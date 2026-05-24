use anyhow::{Context, Result};
use base64::Engine;
use bincode;
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    signature::{Keypair, Signer as SolanaSigner, Signature},
    transaction::Transaction,
};
use spl_associated_token_account::get_associated_token_address;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

const USDC_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
const CONFIRM_POLL_MS: u64 = 500;
const CONFIRM_MAX_POLLS: u32 = 60; // 30 seconds total

pub struct Executor {
    rpc: Arc<RpcClient>,
    keypair: Keypair,
}

impl Executor {
    pub fn new(rpc_url: &str, keypair_path: &str) -> Result<Self> {
        let expanded = if keypair_path.starts_with("~/") {
            if let Some(home) = std::env::var("HOME").ok() {
                format!("{}{}", home, &keypair_path[1..])
            } else {
                keypair_path.to_string()
            }
        } else {
            keypair_path.to_string()
        };

        let path = Path::new(&expanded);
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read keypair from {}", expanded))?;

        let keypair: Keypair = if raw.trim_start().starts_with('[') {
            let bytes: Vec<u8> = serde_json::from_str(&raw)
                .context("keypair JSON parse failed")?;
            Keypair::try_from(&bytes[..])
                .context("failed to create keypair from bytes")?
        } else {
            let bytes = bs58::decode(raw.trim()).into_vec()
                .context("failed to decode bs58 keypair")?;
            Keypair::try_from(&bytes[..])
                .context("failed to create keypair from bs58 bytes")?
        };

        let rpc = Arc::new(RpcClient::new_with_commitment(
            rpc_url.to_string(),
            CommitmentConfig::confirmed(),
        ));

        let pubkey = keypair.pubkey();
        info!("Executor initialized: wallet = {}", pubkey);

        Ok(Self { rpc, keypair })
    }

    pub fn wallet_pubkey(&self) -> String {
        self.keypair.pubkey().to_string()
    }

    /// Get SOL balance in lamports.
    pub fn get_balance(&self) -> Result<u64> {
        Ok(self.rpc.get_balance(&self.keypair.pubkey())?)
    }

    /// Get USDC (SPL token) balance in UI units (e.g. 100.5 USDC).
    /// Returns 0.0 if no token account exists.
    pub fn get_usdc_balance(&self) -> Result<f64> {
        let usdc_mint: solana_sdk::pubkey::Pubkey = USDC_MINT
            .parse()
            .context("invalid USDC mint address")?;
        let ata = get_associated_token_address(&self.keypair.pubkey(), &usdc_mint);

        match self.rpc.get_token_account_balance(&ata) {
            Ok(balance) => {
                let ui_amount = balance.ui_amount.unwrap_or(0.0);
                debug!("USDC balance: {:.6}", ui_amount);
                Ok(ui_amount)
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("could not find account") || msg.contains("Invalid param") {
                    debug!("No USDC token account found, balance = 0");
                    Ok(0.0)
                } else {
                    Err(e).context("failed to get USDC token balance")
                }
            }
        }
    }

    /// Sign a base64-encoded transaction with a FRESH blockhash, then send.
    pub async fn sign_and_send(&self, tx_base64: &str) -> Result<Signature> {
        let tx_bytes = base64::engine::general_purpose::STANDARD
            .decode(tx_base64)
            .context("failed to decode base64 transaction")?;

        let mut tx: Transaction = bincode::deserialize(&tx_bytes)
            .context("failed to deserialize transaction")?;

        // Fetch a fresh blockhash via spawn_blocking (RPC calls are synchronous)
        let rpc = self.rpc.clone();
        let recent_blockhash = tokio::task::spawn_blocking(move || {
            rpc.get_latest_blockhash()
        })
        .await
        .context("spawn_blocking panicked")?
        .context("failed to get latest blockhash")?;

        debug!("Signing with fresh blockhash: {}", recent_blockhash);

        tx.sign(&[&self.keypair], recent_blockhash);

        let serialized = bincode::serialize(&tx).context("failed to serialize signed tx")?;
        let rpc = self.rpc.clone();
        let sig: Result<Signature, anyhow::Error> = tokio::task::spawn_blocking(move || {
            let tx: Transaction = bincode::deserialize(&serialized)
                .map_err(|e| anyhow::anyhow!("deserialize: {}", e))?;
            let sig = rpc.send_transaction_with_config(
                &tx,
                solana_client::rpc_config::RpcSendTransactionConfig {
                    skip_preflight: true,
                    max_retries: Some(3),
                    ..Default::default()
                },
            )
            .map_err(|e| anyhow::anyhow!("send_transaction: {}", e))?;
            Ok(sig)
        })
        .await
        .context("spawn_blocking panicked")?;

        let sig = sig?;
        info!("Transaction sent: {}", sig);

        // Poll for confirmation
        let rpc = self.rpc.clone();
        tokio::task::spawn_blocking(move || {
            for _ in 0..CONFIRM_MAX_POLLS {
                if rpc.confirm_transaction(&sig).unwrap_or(false) {
                    info!("Transaction confirmed: {}", sig);
                    return Ok::<Signature, anyhow::Error>(sig);
                }
                std::thread::sleep(std::time::Duration::from_millis(CONFIRM_POLL_MS));
            }
            warn!("Transaction not confirmed within timeout: {}", sig);
            Ok(sig)
        })
        .await
        .context("spawn_blocking panicked")?
    }

    pub async fn sign_and_send_with_retry(
        &self,
        tx_base64: &str,
        max_retries: u32,
    ) -> Result<Signature> {
        let mut last_err = None;
        for attempt in 0..=max_retries {
            match self.sign_and_send(tx_base64).await {
                Ok(sig) => return Ok(sig),
                Err(e) => {
                    warn!("Attempt {}/{} failed: {:#}", attempt + 1, max_retries + 1, e);
                    last_err = Some(e);
                    if attempt < max_retries {
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown error")))
    }
}
