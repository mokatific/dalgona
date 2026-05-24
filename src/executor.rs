use anyhow::{Context, Result};
use base64::Engine;
use bincode;
use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    signature::{Keypair, Signature, Signer as SolanaSigner},
    transaction::Transaction,
};
use std::path::Path;
use tracing::{debug, info, warn};

pub struct Executor {
    rpc: RpcClient,
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

        // Try JSON array first (standard Solana keypair format)
        let keypair: Keypair = if raw.trim_start().starts_with('[') {
            let bytes: Vec<u8> = serde_json::from_str(&raw).context("keypair JSON parse failed")?;
            Keypair::try_from(&bytes[..]).context("failed to create keypair from bytes")?
        } else {
            // Try bs58 encoded
            let bytes = bs58::decode(raw.trim())
                .into_vec()
                .context("failed to decode bs58 keypair")?;
            Keypair::try_from(&bytes[..]).context("failed to create keypair from bs58 bytes")?
        };

        let rpc =
            RpcClient::new_with_commitment(rpc_url.to_string(), CommitmentConfig::confirmed());

        let pubkey = keypair.pubkey();
        info!("Executor initialized: wallet = {}", pubkey);

        Ok(Self { rpc, keypair })
    }

    pub fn wallet_pubkey(&self) -> String {
        self.keypair.pubkey().to_string()
    }

    pub fn keypair(&self) -> &Keypair {
        &self.keypair
    }

    pub async fn sign_and_send(&self, tx_base64: &str) -> Result<Signature> {
        let tx_bytes = base64::engine::general_purpose::STANDARD
            .decode(tx_base64)
            .context("failed to decode base64 transaction")?;

        let mut tx: Transaction =
            bincode::deserialize(&tx_bytes).context("failed to deserialize transaction")?;

        debug!("Transaction deserialized, signing...");

        tx.sign(&[&self.keypair], tx.message.recent_blockhash);

        let sig = self
            .rpc
            .send_transaction_with_config(
                &tx,
                solana_client::rpc_config::RpcSendTransactionConfig {
                    skip_preflight: true,
                    max_retries: Some(3),
                    ..Default::default()
                },
            )
            .context("failed to send transaction")?;

        info!("Transaction sent: {}", sig);

        // Poll for confirmation
        for _ in 0..30 {
            if self.rpc.confirm_transaction(&sig).unwrap_or(false) {
                info!("Transaction confirmed: {}", sig);
                return Ok(sig);
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        warn!("Transaction not confirmed within timeout: {}", sig);
        Ok(sig)
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
                    warn!(
                        "Attempt {}/{} failed: {:#}",
                        attempt + 1,
                        max_retries + 1,
                        e
                    );
                    last_err = Some(e);
                    if attempt < max_retries {
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown error")))
    }

    pub fn get_balance(&self) -> Result<u64> {
        let balance = self.rpc.get_balance(&self.keypair.pubkey())?;
        Ok(balance)
    }
}
