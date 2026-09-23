// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//! Greedy token-id oracle for the Rust→C++ parity harness (TEST-ONLY; not part of the product).
//!
//! Loads a catalog model on the CPU backend, applies the chat template to one user turn,
//! greedy-decodes up to `max_new` tokens, and prints five tab-separated lines:
//!   model, backend, prompt_ids, output_ids, output_text
//! `cpp/tests/parity/greedy_parity.sh` diffs the `prompt_ids`/`output_ids` lines against the
//! C++ tool that ships with the same contract (sub-project 1b).
//!
//! Usage: cargo run --release -p sapient-generate --example greedy_ids -- <model-alias> "<prompt>" <max_new>

use sapient_generate::{GenerationBackend, LoadOptions, Pipeline, SamplingStrategy};
use sapient_tokenizers::ChatMessage;

fn join(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: greedy_ids <model-alias> <prompt> <max_new>");
        std::process::exit(2);
    }
    let (model, prompt) = (&args[0], &args[1]);
    let max_new: usize = args[2].parse()?;

    let mut opts = LoadOptions {
        backend: GenerationBackend::Cpu,
        ..LoadOptions::default()
    };
    opts.hub.quiet = true;
    let pipeline = Pipeline::from_pretrained_with_opts(model, opts).await?;

    let text = pipeline.format_chat_prompt(&[ChatMessage::user(prompt.clone())])?;
    let prompt_ids = pipeline.tokenizer().encode(&text)?;
    let stop_ids = pipeline.eos_token_ids_pub();
    let out =
        pipeline.generate_token_ids(&prompt_ids, max_new, &stop_ids, SamplingStrategy::Greedy)?;
    let decoded = pipeline.tokenizer().decode(&out, true)?;

    println!("model\t{model}");
    println!("backend\t{}", pipeline.backend_display_label());
    println!("prompt_ids\t{}", join(&prompt_ids));
    println!("output_ids\t{}", join(&out));
    println!("output_text\t{}", escape(&decoded));
    Ok(())
}
