# Copyright (c) 2026 Angela Louise Trainor
# MIT License

import torch

class VisualCortexBENNTester:
    """
    Simple diagnostic tester for the VisualCortexBENN model.

    Usage:
        tester = VisualCortexBENNTester()
        tester.run_all_tests()
    """

    def __init__(self, model_class, device="cpu"):
        self.device = device
        self.model = model_class().to(device)
        self.model.eval()

    def generate_test_input(self, batch_size=2, height=64, width=64):
        """
        Generates a random RGB image batch.
        """
        return torch.randn(batch_size, 3, height, width).to(self.device)

    def test_forward_pass(self):
        """
        Tests whether the model can process an input image batch.
        """
        x = self.generate_test_input()

        with torch.no_grad():
            outputs = self.model(x)

        assert isinstance(outputs, dict), "Output must be a dictionary."

        required_keys = [
            "color",
            "texture",
            "shape_emotion",
            "centroid",
            "entities",
            "attention"
        ]

        for key in required_keys:
            assert key in outputs, f"Missing output key: {key}"

        print("✓ Forward pass successful")
        return outputs

    def test_output_shapes(self):
        """
        Confirms the model returns expected tensor shapes.
        """
        x = self.generate_test_input()
        outputs = self.model(x)

        batch = x.size(0)

        expected_shapes = {
            "color": (batch, 64),
            "texture": (batch, 64),
            "shape_emotion": (batch, 8),
            "centroid": (batch, 32),
            "entities": (batch, 128),
            "attention": (batch, 1)
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(outputs[key].shape)
            assert actual_shape == expected_shape, \
                f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}"

        print("✓ Output shapes are correct")

    def test_backward_pass(self):
        """
        Ensures gradients flow properly.
        """
        self.model.train()

        x = self.generate_test_input()
        outputs = self.model(x)

        loss = 0
        for v in outputs.values():
            loss += v.mean()

        loss.backward()

        grad_exists = any(p.grad is not None for p in self.model.parameters())
        assert grad_exists, "No gradients detected."

        print("✓ Backward pass successful")

    def run_all_tests(self):
        """
        Runs the full diagnostic suite.
        """
        print("Running VisualCortexBENN diagnostics...\n")

        self.test_forward_pass()
        self.test_output_shapes()
        self.test_backward_pass()

        print("\n✓ All tests passed successfully.")
