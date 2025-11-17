#!/usr/bin/env python3
"""
Test to debug the ABMIL model shape issue
"""

import torch
import torch.nn as nn
from src.builder import create_model

print("="*70)
print("ABMIL MODEL SHAPE DEBUG")
print("="*70)

# Create model
print("\n1. Creating ABMIL model...")
model = create_model(
    'abmil.base.uni_v2.pc108-24k',
    num_classes=3,
    dropout=0.2,
    gate=True
)
model.eval()
print("   Model created")

# Test with different input shapes
test_cases = [
    ("Single sample (batch_size=1)", torch.randn(1, 500, 1536)),
    ("Batch of 2 samples", torch.randn(2, 512, 1536)),
    ("Batch of 4 samples", torch.randn(4, 512, 1536)),
]

for name, features in test_cases:
    print(f"\n2. Testing: {name}")
    print(f"   Input shape: {features.shape}")

    try:
        with torch.no_grad():
            # Add debug hooks to see intermediate shapes
            intermediate_shapes = {}

            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        intermediate_shapes[name] = tuple(o.shape if isinstance(o, torch.Tensor) else None for o in output)
                    else:
                        intermediate_shapes[name] = output.shape if isinstance(output, torch.Tensor) else None
                return hook

            # Register hooks
            model.patch_embed.register_forward_hook(hook_fn('patch_embed'))
            model.global_attn.register_forward_hook(hook_fn('global_attn'))

            # Forward pass
            results_dict, log_dict = model(features)
            logits = results_dict['logits']

            print(f"   ✓ Success!")
            print(f"   Output logits shape: {logits.shape}")
            print(f"   Intermediate shapes:")
            for k, v in intermediate_shapes.items():
                print(f"      {k}: {v}")

    except Exception as e:
        print(f"   ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Test with actual validation scenario
print("\n" + "="*70)
print("TESTING WITH ACTUAL VALIDATION SCENARIO")
print("="*70)

# Simulate validation with criterion and labels
criterion = nn.CrossEntropyLoss()

print("\n3. Testing with loss_fn and label...")
features = torch.randn(1, 500, 1536)
labels = torch.tensor([0])

print(f"   Input shape: {features.shape}")
print(f"   Labels: {labels}")

try:
    with torch.no_grad():
        results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
        logits = results_dict['logits']
        loss = results_dict['loss']

        print(f"   ✓ Success!")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Loss: {loss.item()}")

except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
