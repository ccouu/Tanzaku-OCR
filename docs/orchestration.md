[Phase 1: Research & Spec] ──> agy (Gemini 6 Argon)
                                    │ (Produces ViT Architecture & Tokenizer Spec)
                                    ▼
[Phase 2: Code & Export]   ──> Terra (Codex 6.1 Terra)
                                    │ (Implements PyTorch, ONNX/CoreML Export)
                                    ▼
[Phase 3: Review & Test]   ──> Flash (DeepSeek 4.1 Flash)
                                    │ (Runs Static Analysis, Shape & Safety Verification)
                                    ▼
[Phase 4: Profile & Polish]──> Astra (Codex 6 Astra)
                                    │ (Calculates FLOPs, Latency & Final Verification)
                                    ▼
                             [Deployment Target: CoreML / TFLite Mobile Package]