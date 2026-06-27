"""Tests for domain expansion: 10-domain taxonomy, subject aliases, and API endpoints.

Covers:
- Taxonomy loads all 10 domains
- Each new domain has correct subjects
- Subject aliases work (solidity, rust, hermes, py, js, ts, k8s, tf)
- GET /api/taxonomy returns all domains
- GET /api/taxonomy/{domain} returns correct subjects
- GET /api/taxonomy/search finds subjects
- resolve_subject works for aliases and full names
- Edge cases: nonexistent domain, empty search, case-insensitive
"""

import os
import sys
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_python_dir = os.path.join(_project_root, "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)


# ═══════════════════════════════════════════════════════════════════
# Taxonomy Loading: 10 Domains
# ═══════════════════════════════════════════════════════════════════

class TestTaxonomyLoadsAllDomains(unittest.TestCase):
    """Verify the taxonomy file loads and contains all 22 domains (10 original + 12 expanded)."""

    def test_taxonomy_loads_without_error(self):
        """load_taxonomy() should succeed without raising."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        self.assertIsInstance(tax, dict)

    def test_taxonomy_has_22_domains(self):
        """Taxonomy should have exactly 22 domains (10 original + 12 new)."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        categories = tax.get("categories", {})
        self.assertEqual(len(categories), 22,
                         f"Expected 22 domains, got {len(categories)}: {sorted(categories.keys())}")

    def test_taxonomy_has_expected_domain_names(self):
        """All 22 expected domain names should be present."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        categories = tax.get("categories", {})
        expected_domains = [
            # Original 10
            "STEM",
            "Humanities",
            "Social Science",
            "Professional",
            "Other",
            "Programming_Languages",
            "Agent_Frameworks",
            "Blockchain_Web3",
            "DevOps_Infrastructure",
            "Security_Cryptography",
            # New 12
            "Advanced_Reasoning",
            "Agentic_Capabilities",
            "Code_Engineering",
            "AI_ML_Knowledge",
            "Multimodal_Understanding",
            "Long_Context_Retrieval",
            "Creative_Tasks",
            "Real_World_Knowledge",
            "Mathematical_Reasoning",
            "Scientific_Reasoning",
            "Natural_Language_Understanding",
            "Instruction_Following_Alignment",
        ]
        for domain in expected_domains:
            self.assertIn(domain, categories,
                         f"Domain '{domain}' not found in taxonomy")

    def test_taxonomy_has_subject_mapping(self):
        """Taxonomy should have a subject_mapping section."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        self.assertIn("subject_mapping", tax)
        self.assertIsInstance(tax["subject_mapping"], dict)
        self.assertGreater(len(tax["subject_mapping"]), 0)


# ═══════════════════════════════════════════════════════════════════
# New Domain Subjects
# ═══════════════════════════════════════════════════════════════════

class TestNewDomainSubjects(unittest.TestCase):
    """Verify each new domain has the correct subjects."""

    def setUp(self):
        from mindforge.probe.question_gen import load_taxonomy
        self.tax = load_taxonomy()
        self.categories = self.tax["categories"]

    def test_programming_languages_has_subjects(self):
        """Programming_Languages should contain key languages."""
        subjects = self.categories["Programming_Languages"]
        for lang in ["python", "javascript", "typescript", "rust", "go", "java",
                      "cpp", "csharp", "ruby", "swift", "kotlin", "haskell"]:
            self.assertIn(lang, subjects,
                         f"Language '{lang}' not in Programming_Languages")

    def test_programming_languages_count(self):
        """Programming_Languages should have at least 10 subjects."""
        subjects = self.categories["Programming_Languages"]
        self.assertGreaterEqual(len(subjects), 10,
                                f"Expected >= 10 subjects, got {len(subjects)}")

    def test_agent_frameworks_has_subjects(self):
        """Agent_Frameworks should contain key agent frameworks."""
        subjects = self.categories["Agent_Frameworks"]
        for fw in ["langchain", "llamaindex", "crewai", "autogen", "hermes_agent"]:
            self.assertIn(fw, subjects,
                         f"Framework '{fw}' not in Agent_Frameworks")

    def test_agent_frameworks_count(self):
        """Agent_Frameworks should have at least 5 subjects."""
        self.assertGreaterEqual(len(self.categories["Agent_Frameworks"]), 5)

    def test_blockchain_web3_has_subjects(self):
        """Blockchain_Web3 should contain key blockchain subjects."""
        subjects = self.categories["Blockchain_Web3"]
        for bc in ["solidity", "web3js", "ethersjs", "hardhat", "foundry", "solana"]:
            self.assertIn(bc, subjects,
                         f"Blockchain subject '{bc}' not in Blockchain_Web3")

    def test_blockchain_web3_count(self):
        """Blockchain_Web3 should have at least 5 subjects."""
        self.assertGreaterEqual(len(self.categories["Blockchain_Web3"]), 5)

    def test_devops_infrastructure_has_subjects(self):
        """DevOps_Infrastructure should contain key DevOps subjects."""
        subjects = self.categories["DevOps_Infrastructure"]
        for devops in ["docker", "kubernetes", "terraform", "ci_cd"]:
            self.assertIn(devops, subjects,
                         f"DevOps subject '{devops}' not in DevOps_Infrastructure")

    def test_devops_infrastructure_count(self):
        """DevOps_Infrastructure should have at least 4 subjects."""
        self.assertGreaterEqual(len(self.categories["DevOps_Infrastructure"]), 4)

    def test_security_cryptography_has_subjects(self):
        """Security_Cryptography should contain key security subjects."""
        subjects = self.categories["Security_Cryptography"]
        for sec in ["cryptography", "network_security", "pentesting", "secure_coding"]:
            self.assertIn(sec, subjects,
                         f"Security subject '{sec}' not in Security_Cryptography")

    def test_security_cryptography_count(self):
        """Security_Cryptography should have at least 3 subjects."""
        self.assertGreaterEqual(len(self.categories["Security_Cryptography"]), 3)

    def test_original_domains_still_intact(self):
        """Original 5 domains (STEM, Humanities, Social Science, Professional, Other) should still have subjects."""
        for domain in ["STEM", "Humanities", "Social Science", "Professional", "Other"]:
            self.assertGreater(len(self.categories[domain]), 0,
                             f"Domain '{domain}' has no subjects")

    def test_stem_has_mathematics(self):
        """STEM should still contain high_school_mathematics."""
        self.assertIn("high_school_mathematics", self.categories["STEM"])

    def test_humanities_has_philosophy(self):
        """Humanities should still contain philosophy."""
        self.assertIn("philosophy", self.categories["Humanities"])

    def test_no_duplicate_subjects_within_domain(self):
        """No domain should have duplicate subjects."""
        for domain, subjects in self.categories.items():
            self.assertEqual(len(subjects), len(set(subjects)),
                           f"Domain '{domain}' has duplicate subjects")


# ═══════════════════════════════════════════════════════════════════
# Expanded Domain Subjects (12 new domains)
# ═══════════════════════════════════════════════════════════════════

class TestExpandedDomainSubjects(unittest.TestCase):
    """Verify each of the 12 new expanded domains has correct subjects."""

    def setUp(self):
        from mindforge.probe.question_gen import load_taxonomy
        self.tax = load_taxonomy()
        self.categories = self.tax["categories"]

    def test_advanced_reasoning_has_subjects(self):
        """Advanced_Reasoning should contain key reasoning subjects."""
        subjects = self.categories["Advanced_Reasoning"]
        for s in ["causal_reasoning", "deductive_reasoning", "abstract_pattern_induction"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Advanced_Reasoning")
        self.assertGreaterEqual(len(subjects), 10)

    def test_agentic_capabilities_has_subjects(self):
        """Agentic_Capabilities should contain key agentic subjects."""
        subjects = self.categories["Agentic_Capabilities"]
        for s in ["tool_selection", "multi_step_planning", "web_navigation"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Agentic_Capabilities")
        self.assertGreaterEqual(len(subjects), 10)

    def test_code_engineering_has_subjects(self):
        """Code_Engineering should contain key engineering subjects."""
        subjects = self.categories["Code_Engineering"]
        for s in ["system_design", "debugging_strategies", "code_review"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Code_Engineering")
        self.assertGreaterEqual(len(subjects), 10)

    def test_ai_ml_knowledge_has_subjects(self):
        """AI_ML_Knowledge should contain key AI/ML subjects."""
        subjects = self.categories["AI_ML_Knowledge"]
        for s in ["transformer_architecture", "training_methodology", "rag_systems"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in AI_ML_Knowledge")
        self.assertGreaterEqual(len(subjects), 10)

    def test_multimodal_understanding_has_subjects(self):
        """Multimodal_Understanding should contain key multimodal subjects."""
        subjects = self.categories["Multimodal_Understanding"]
        for s in ["image_understanding", "video_understanding", "cross_modal_reasoning"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Multimodal_Understanding")
        self.assertGreaterEqual(len(subjects), 8)

    def test_long_context_retrieval_has_subjects(self):
        """Long_Context_Retrieval should contain key retrieval subjects."""
        subjects = self.categories["Long_Context_Retrieval"]
        for s in ["information_retrieval", "needle_in_haystack", "fact_verification"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Long_Context_Retrieval")
        self.assertGreaterEqual(len(subjects), 8)

    def test_creative_tasks_has_subjects(self):
        """Creative_Tasks should contain key creative subjects."""
        subjects = self.categories["Creative_Tasks"]
        for s in ["storytelling", "poetry_composition", "dialogue_writing"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Creative_Tasks")
        self.assertGreaterEqual(len(subjects), 8)

    def test_real_world_knowledge_has_subjects(self):
        """Real_World_Knowledge should contain key real-world subjects."""
        subjects = self.categories["Real_World_Knowledge"]
        for s in ["current_events", "pop_culture", "sports_knowledge"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Real_World_Knowledge")
        self.assertGreaterEqual(len(subjects), 8)

    def test_mathematical_reasoning_has_subjects(self):
        """Mathematical_Reasoning should contain key math subjects."""
        subjects = self.categories["Mathematical_Reasoning"]
        for s in ["algebraic_reasoning", "competition_mathematics", "linear_algebra"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Mathematical_Reasoning")
        self.assertGreaterEqual(len(subjects), 10)

    def test_scientific_reasoning_has_subjects(self):
        """Scientific_Reasoning should contain key science subjects."""
        subjects = self.categories["Scientific_Reasoning"]
        for s in ["experimental_design", "hypothesis_testing", "statistical_inference"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Scientific_Reasoning")
        self.assertGreaterEqual(len(subjects), 8)

    def test_natural_language_understanding_has_subjects(self):
        """Natural_Language_Understanding should contain key NLU subjects."""
        subjects = self.categories["Natural_Language_Understanding"]
        for s in ["reading_comprehension", "natural_language_inference", "pragmatics"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Natural_Language_Understanding")
        self.assertGreaterEqual(len(subjects), 8)

    def test_instruction_following_alignment_has_subjects(self):
        """Instruction_Following_Alignment should contain key alignment subjects."""
        subjects = self.categories["Instruction_Following_Alignment"]
        for s in ["format_compliance", "safety_guidelines", "preference_alignment"]:
            self.assertIn(s, subjects, f"Subject '{s}' not in Instruction_Following_Alignment")
        self.assertGreaterEqual(len(subjects), 5)

    def test_subject_descriptions_section_exists(self):
        """Taxonomy should have a subject_descriptions section for new subjects."""
        self.assertIn("subject_descriptions", self.tax)
        self.assertIsInstance(self.tax["subject_descriptions"], dict)
        self.assertGreater(len(self.tax["subject_descriptions"]), 100)

    def test_all_new_subjects_have_descriptions(self):
        """Every new domain subject should have a description."""
        original_domains = {"STEM", "Humanities", "Social Science", "Professional", "Other"}
        new_subjects = set()
        for domain, subjects in self.categories.items():
            if domain not in original_domains and domain not in {
                "Programming_Languages", "Agent_Frameworks", "Blockchain_Web3",
                "DevOps_Infrastructure", "Security_Cryptography"
            }:
                new_subjects.update(subjects)
        descriptions = self.tax.get("subject_descriptions", {})
        missing = new_subjects - set(descriptions.keys())
        self.assertEqual(len(missing), 0,
                        f"Subjects without descriptions: {missing}")


# ═══════════════════════════════════════════════════════════════════
# Subject Alias Tests
# ═══════════════════════════════════════════════════════════════════

class TestSubjectAliases(unittest.TestCase):
    """Verify subject aliases resolve correctly via resolve_subject()."""

    def test_alias_solidity(self):
        """Alias 'solidity' should resolve to 'solidity'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("solidity"), "solidity")

    def test_alias_rust(self):
        """Alias 'rust' should resolve to 'rust'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("rust"), "rust")

    def test_alias_hermes(self):
        """Alias 'hermes' should resolve to 'hermes_agent'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("hermes"), "hermes_agent")

    def test_alias_py(self):
        """Alias 'py' should resolve to 'python'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("py"), "python")

    def test_alias_js(self):
        """Alias 'js' should resolve to 'javascript'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("js"), "javascript")

    def test_alias_ts(self):
        """Alias 'ts' should resolve to 'typescript'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("ts"), "typescript")

    def test_alias_k8s(self):
        """Alias 'k8s' should resolve to 'kubernetes'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("k8s"), "kubernetes")

    def test_alias_tf(self):
        """Alias 'tf' should resolve to 'terraform'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("tf"), "terraform")

    def test_alias_math(self):
        """Alias 'math' should resolve to 'high_school_mathematics'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("math"), "high_school_mathematics")

    def test_alias_ml(self):
        """Alias 'ml' should resolve to 'machine_learning'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("ml"), "machine_learning")

    def test_alias_cs(self):
        """Alias 'cs' should resolve to 'high_school_computer_science'."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("cs"), "high_school_computer_science")

    def test_resolve_full_subject_name(self):
        """Full subject names should resolve to themselves."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertEqual(resolve_subject("python"), "python")
        self.assertEqual(resolve_subject("rust"), "rust")
        self.assertEqual(resolve_subject("solidity"), "solidity")
        self.assertEqual(resolve_subject("hermes_agent"), "hermes_agent")

    def test_resolve_invalid_subject_returns_none(self):
        """Invalid subject should return None."""
        from mindforge.probe.question_gen import resolve_subject
        self.assertIsNone(resolve_subject("totally_fake_subject_xyz123"))

    def test_resolve_subject_is_case_insensitive_for_spaces(self):
        """resolve_subject should handle subjects with spaces converted to underscores."""
        from mindforge.probe.question_gen import resolve_subject
        # These should work because resolve_subject tries replacing spaces with underscores
        result = resolve_subject("high school mathematics")
        # Should resolve to high_school_mathematics
        self.assertIsNotNone(result)


# ═══════════════════════════════════════════════════════════════════
# FastAPI: GET /api/taxonomy
# ═══════════════════════════════════════════════════════════════════

class TestAPITaxonomy(unittest.TestCase):
    """Tests for GET /api/taxonomy endpoint."""

    @classmethod
    def setUpClass(cls):
        try:
            from starlette.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError:
            cls.client = None

    def setUp(self):
        if self.client is None:
            self.skipTest("FastAPI/Starlette TestClient not available")

    def test_taxonomy_returns_200(self):
        """GET /api/taxonomy should return 200."""
        resp = self.client.get("/api/taxonomy")
        self.assertEqual(resp.status_code, 200)

    def test_taxonomy_returns_all_22_domains(self):
        """GET /api/taxonomy should return all 22 domains (10 original + 12 new)."""
        resp = self.client.get("/api/taxonomy")
        data = resp.json()
        categories = data.get("categories", {})
        self.assertEqual(len(categories), 22,
                         f"Expected 22 domains, got {len(categories)}")

    def test_taxonomy_has_expected_domain_keys(self):
        """GET /api/taxonomy should include all expected domain names."""
        resp = self.client.get("/api/taxonomy")
        categories = resp.json().get("categories", {})
        for domain in ["STEM", "Humanities", "Programming_Languages",
                        "Agent_Frameworks", "Blockchain_Web3",
                        "DevOps_Infrastructure", "Security_Cryptography"]:
            self.assertIn(domain, categories,
                         f"Domain '{domain}' not in API response")

    def test_taxonomy_has_subject_mapping(self):
        """GET /api/taxonomy should include subject_mapping."""
        resp = self.client.get("/api/taxonomy")
        data = resp.json()
        self.assertIn("subject_mapping", data)
        self.assertGreater(len(data["subject_mapping"]), 0)

    def test_taxonomy_has_new_aliases_in_mapping(self):
        """GET /api/taxonomy should include the new aliases in subject_mapping."""
        resp = self.client.get("/api/taxonomy")
        mapping = resp.json().get("subject_mapping", {})
        for alias in ["solidity", "rust", "hermes", "py", "js", "ts", "k8s", "tf"]:
            self.assertIn(alias, mapping,
                         f"Alias '{alias}' not in subject_mapping")

    def test_taxonomy_programming_languages_has_subjects(self):
        """GET /api/taxonomy should show subjects in Programming_Languages."""
        resp = self.client.get("/api/taxonomy")
        categories = resp.json().get("categories", {})
        subjects = categories.get("Programming_Languages", [])
        self.assertIn("python", subjects)
        self.assertIn("rust", subjects)
        self.assertIn("typescript", subjects)


# ═══════════════════════════════════════════════════════════════════
# FastAPI: GET /api/taxonomy/{domain}
# ═══════════════════════════════════════════════════════════════════

class TestAPITaxonomyByDomain(unittest.TestCase):
    """Tests for GET /api/taxonomy/{domain} endpoint."""

    @classmethod
    def setUpClass(cls):
        try:
            from starlette.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError:
            cls.client = None

    def setUp(self):
        if self.client is None:
            self.skipTest("FastAPI/Starlette TestClient not available")

    def test_get_stem_domain_returns_200(self):
        """GET /api/taxonomy/STEM should return 200."""
        resp = self.client.get("/api/taxonomy/STEM")
        self.assertEqual(resp.status_code, 200)

    def test_get_stem_domain_has_subjects(self):
        """GET /api/taxonomy/STEM should return subjects list."""
        resp = self.client.get("/api/taxonomy/STEM")
        data = resp.json()
        self.assertIn("subjects", data)
        self.assertIn("count", data)
        self.assertGreater(data["count"], 0)

    def test_get_programming_languages_domain(self):
        """GET /api/taxonomy/Programming_Languages should return language subjects."""
        resp = self.client.get("/api/taxonomy/Programming_Languages")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("python", data["subjects"])
        self.assertIn("rust", data["subjects"])

    def test_get_blockchain_web3_domain(self):
        """GET /api/taxonomy/Blockchain_Web3 should return blockchain subjects."""
        resp = self.client.get("/api/taxonomy/Blockchain_Web3")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("solidity", data["subjects"])

    def test_get_agent_frameworks_domain(self):
        """GET /api/taxonomy/Agent_Frameworks should return framework subjects."""
        resp = self.client.get("/api/taxonomy/Agent_Frameworks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("langchain", data["subjects"])

    def test_get_devops_domain(self):
        """GET /api/taxonomy/DevOps_Infrastructure should return DevOps subjects."""
        resp = self.client.get("/api/taxonomy/DevOps_Infrastructure")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("docker", data["subjects"])
        self.assertIn("kubernetes", data["subjects"])

    def test_get_security_domain(self):
        """GET /api/taxonomy/Security_Cryptography should return security subjects."""
        resp = self.client.get("/api/taxonomy/Security_Cryptography")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cryptography", data["subjects"])

    def test_domain_response_has_correct_structure(self):
        """GET /api/taxonomy/{domain} response should have domain, subjects, count."""
        resp = self.client.get("/api/taxonomy/STEM")
        data = resp.json()
        self.assertEqual(data["domain"], "STEM")
        self.assertIsInstance(data["subjects"], list)
        self.assertIsInstance(data["count"], int)
        self.assertEqual(data["count"], len(data["subjects"]))

    def test_nonexistent_domain_returns_404(self):
        """GET /api/taxonomy/Nonexistent should return 404."""
        resp = self.client.get("/api/taxonomy/Nonexistent_Domain")
        self.assertEqual(resp.status_code, 404)

    def test_domain_case_insensitive(self):
        """GET /api/taxonomy/stem should work (case-insensitive)."""
        resp = self.client.get("/api/taxonomy/stem")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["domain"], "STEM")


# ═══════════════════════════════════════════════════════════════════
# FastAPI: GET /api/taxonomy/search
# ═══════════════════════════════════════════════════════════════════

class TestAPITaxonomySearch(unittest.TestCase):
    """Tests for GET /api/taxonomy/search endpoint."""

    @classmethod
    def setUpClass(cls):
        try:
            from starlette.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError:
            cls.client = None

    def setUp(self):
        if self.client is None:
            self.skipTest("FastAPI/Starlette TestClient not available")

    def test_search_python_returns_results(self):
        """Search for 'python' should find the python subject."""
        resp = self.client.get("/api/taxonomy/search?q=python")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)
        subjects = [r["subject"] for r in data["results"]]
        self.assertIn("python", subjects)

    def test_search_solidity_returns_results(self):
        """Search for 'solidity' should find the solidity subject."""
        resp = self.client.get("/api/taxonomy/search?q=solidity")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)

    def test_search_rust_returns_results(self):
        """Search for 'rust' should find the rust subject."""
        resp = self.client.get("/api/taxonomy/search?q=rust")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)

    def test_search_kubernetes_returns_results(self):
        """Search for 'kubernetes' should find the kubernetes subject."""
        resp = self.client.get("/api/taxonomy/search?q=kubernetes")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)

    def test_search_returns_domain_info(self):
        """Search results should include domain information."""
        resp = self.client.get("/api/taxonomy/search?q=python")
        data = resp.json()
        for result in data["results"]:
            self.assertIn("domain", result)
            self.assertIn("subject", result)

    def test_search_empty_query_returns_400(self):
        """Empty search query should return 400."""
        resp = self.client.get("/api/taxonomy/search?q=")
        self.assertEqual(resp.status_code, 400)

    def test_search_no_results(self):
        """Search for nonexistent subject should return 0 results."""
        resp = self.client.get("/api/taxonomy/search?q=nonexistent_subject_xyz123")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 0)

    def test_search_partial_match(self):
        """Search should match partial subject names."""
        resp = self.client.get("/api/taxonomy/search?q=crypto")
        data = resp.json()
        # Should find 'cryptography' at minimum
        subjects = [r["subject"] for r in data["results"]]
        self.assertIn("cryptography", subjects)

    def test_search_case_insensitive(self):
        """Search should be case-insensitive."""
        resp_lower = self.client.get("/api/taxonomy/search?q=python")
        resp_upper = self.client.get("/api/taxonomy/search?q=PYTHON")
        self.assertEqual(resp_lower.json()["count"], resp_upper.json()["count"])

    def test_search_math_finds_mathematics(self):
        """Search for 'math' should find mathematics-related subjects."""
        resp = self.client.get("/api/taxonomy/search?q=math")
        data = resp.json()
        subjects = [r["subject"] for r in data["results"]]
        # Should find at least high_school_mathematics
        self.assertTrue(any("math" in s.lower() for s in subjects),
                       f"No math subjects found in search results: {subjects}")

    def test_search_security_finds_multiple(self):
        """Search for 'security' should find multiple security subjects."""
        resp = self.client.get("/api/taxonomy/search?q=security")
        data = resp.json()
        # Should find network_security, computer_security, security_studies, secure_coding
        self.assertGreaterEqual(data["count"], 2,
                               f"Expected >= 2 security results, got {data['count']}")

    def test_search_response_has_query_echo(self):
        """Search response should echo the query."""
        resp = self.client.get("/api/taxonomy/search?q=python")
        data = resp.json()
        self.assertEqual(data["query"], "python")


# ═══════════════════════════════════════════════════════════════════
# Cross-Domain Integration
# ═══════════════════════════════════════════════════════════════════

class TestCrossDomainIntegration(unittest.TestCase):
    """Integration tests crossing taxonomy, resolve_subject, and API."""

    def test_all_resolved_subjects_exist_in_taxonomy(self):
        """Every alias in subject_mapping should resolve to a subject that exists in categories."""
        from mindforge.probe.question_gen import load_taxonomy, resolve_subject
        tax = load_taxonomy()
        all_subjects = set()
        for cat_subjects in tax["categories"].values():
            all_subjects.update(cat_subjects)

        for alias in tax["subject_mapping"]:
            resolved = resolve_subject(alias)
            self.assertIsNotNone(resolved,
                               f"Alias '{alias}' resolved to None")
            self.assertIn(resolved, all_subjects,
                         f"Alias '{alias}' resolved to '{resolved}' which is not in any domain")

    def test_total_subject_count_across_domains(self):
        """Total subjects across all domains should be substantial (257 after expansion)."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        total = sum(len(subjects) for subjects in tax["categories"].values())
        self.assertGreater(total, 200,
                          f"Expected > 200 total subjects, got {total}")

    def test_no_subject_in_multiple_domains(self):
        """No subject should appear in more than one domain."""
        from mindforge.probe.question_gen import load_taxonomy
        tax = load_taxonomy()
        all_subjects = []
        for domain, subjects in tax["categories"].items():
            all_subjects.extend(subjects)

        duplicates = [s for s in all_subjects if all_subjects.count(s) > 1]
        self.assertEqual(len(duplicates), 0,
                        f"Duplicate subjects found across domains: {set(duplicates)}")


if __name__ == "__main__":
    unittest.main()
