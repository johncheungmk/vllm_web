import unittest

from app.main import (
    VllmProfile,
    build_command,
    command_for_shell,
    data_parallel_node_commands,
    validation_report,
)


class CommandGenerationTests(unittest.TestCase):
    def test_eagle3_can_be_configured_without_aux_model(self) -> None:
        profile = VllmProfile(speculative_method="eagle3", speculative_model="")

        report = validation_report(profile)
        shell = command_for_shell(build_command(profile))

        self.assertEqual([], report["errors"])
        self.assertIn('"method":"eagle3"', shell)
        self.assertNotIn('"model"', shell)

    def test_draft_model_requires_aux_model(self) -> None:
        profile = VllmProfile(speculative_method="draft_model", speculative_model="")

        report = validation_report(profile)

        self.assertIn("draft_model speculative decoding requires a compatible auxiliary model.", report["errors"])

    def test_custom_generation_config_requires_path(self) -> None:
        profile = VllmProfile(generation_config="custom", generation_config_path="")

        report = validation_report(profile)

        self.assertIn("Generation config custom path is required when Generation config is set to custom path.", report["errors"])

    def test_data_parallel_generates_per_node_commands(self) -> None:
        profile = VllmProfile(
            deployment_mode="data_parallel_internal",
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_address="10.0.0.10",
            data_parallel_rpc_port=13345,
            ray_node_ips="10.0.0.10\n10.0.0.11",
        )

        commands = data_parallel_node_commands(profile)
        joined = "\n".join(commands)

        self.assertIn("# Node 0 (10.0.0.10)", joined)
        self.assertIn("# Node 1 (10.0.0.11)", joined)
        self.assertIn("--data-parallel-size 4", joined)
        self.assertIn("--data-parallel-start-rank 2", joined)
        self.assertIn("--headless", joined)


if __name__ == "__main__":
    unittest.main()
