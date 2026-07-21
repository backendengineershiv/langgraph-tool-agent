import unittest


class EnvironmentConfigTests(unittest.TestCase):
    def test_graph_module_reads_required_openai_settings(self):
        from app import graph

        self.assertTrue(graph.os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        self.assertTrue(graph.os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.assertTrue(graph.os.getenv("AZURE_OPENAI_API_KEY"))


if __name__ == "__main__":
    unittest.main()
