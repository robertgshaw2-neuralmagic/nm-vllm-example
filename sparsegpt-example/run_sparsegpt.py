import sparseml.transformers

original_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
calibration_dataset = "open_platypus"
output_directory = "tinyllama-pruned"

recipe = """
test_stage:
  obcq_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      sequential_update: false
      targets: ['re:model.layers.\\d*$']
"""

# Apply SparseGPT to the model
sparseml.transformers.oneshot(
    model=original_model_name,
    dataset=calibration_dataset,
    recipe=recipe,
    output_dir=output_directory,
)