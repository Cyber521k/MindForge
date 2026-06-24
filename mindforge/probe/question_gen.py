"""Question generation for probing using MMLU dataset.

Phase 2 additions:
    - Tier 2 (depth): generate_tier2_followups(question, model_answer, subject)
    - Tier 3 (edge cases): generate_tier3_edge_cases(subject)
    - ProbeEngine integration for multi-tier probing
"""

import os
import yaml
import logging
import re

logger = logging.getLogger(__name__)

# Path to taxonomy YAML
TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "taxonomy",
    "subjects.yaml",
)

_LETTERS = ["A", "B", "C", "D"]

# Set of original MMLU subjects (for fast lookup)
def _get_mmlu_subjects():
    """Return the set of original MMLU subject keys."""
    taxonomy = load_taxonomy()
    mmlu_subjects = set()
    # Original MMLU categories (not the new custom ones)
    mmlu_categories = {"STEM", "Humanities", "Social Science", "Professional", "Other"}
    for cat, subjects in taxonomy.get("categories", {}).items():
        if cat in mmlu_categories:
            mmlu_subjects.update(subjects)
    return mmlu_subjects


# Descriptions for non-MMLU (custom) subjects. Used to generate MCQ questions.
SUBJECT_DESCRIPTIONS = {
    # Agent Frameworks
    "hermes_agent": "Hermes Agent is an autonomous AI agent framework by Nous Research that uses tools, skills, memory, and multi-provider LLM support. It features a CLI interface, terminal execution, web browsing, and persistent memory across sessions.",
    "pi_agent": "Pi Agent is a personal intelligence agent framework designed for conversational AI assistance, task automation, and integration with external APIs and services.",
    "openclaw": "OpenClaw is an open-source AI agent framework focused on extensibility, plugin architectures, and multi-agent orchestration with a focus on developer productivity.",
    "clawcode": "ClawCode is a code-generation-focused AI agent that integrates with IDEs and version control systems to assist with software development workflows.",
    "autogpt": "AutoGPT is an autonomous AI agent that chains LLM calls to achieve complex goals. It uses a plan-act-reflect loop with memory, file operations, and web browsing capabilities.",
    "crewai": "CrewAI is a multi-agent orchestration framework where specialized agents collaborate on tasks. It supports role-based agents, sequential and hierarchical task delegation, and tool integration.",
    "langchain": "LangChain is a framework for building LLM applications with composable chains, agents, memory, retrievers, and tools. It supports document loaders, text splitters, vector stores, and output parsers.",
    "llamaindex": "LlamaIndex is a data framework for LLM applications focused on data ingestion, indexing, and retrieval. It provides document readers, index structures, query engines, and retrieval-augmented generation pipelines.",
    "autogen": "AutoGen by Microsoft is a multi-agent conversation framework where agents communicate via messages. It supports code execution, group chats, and human-in-the-loop workflows.",
    "metagpt": "MetaGPT is a multi-agent framework that simulates a software company with roles like Product Manager, Architect, and Engineer. Agents collaborate following SOPs to produce software artifacts.",
    "chatdev": "ChatDev is a conversational software development framework where AI agents act as a development team. It models the software development lifecycle as a communication protocol between agents.",
    "babyagi": "BabyAGI is a task-driven autonomous agent that creates, prioritizes, and executes tasks based on an objective. It uses a task creation loop, task list, and result storage to work toward goals iteratively.",

    # Programming Languages
    "python": "Python is a high-level, dynamically typed programming language known for readability and simplicity. It supports multiple paradigms (OOP, functional, procedural), has a rich standard library, and uses indentation for code blocks. Key features include list comprehensions, decorators, generators, context managers, and a GIL for thread safety.",
    "javascript": "JavaScript is a dynamic, multi-paradigm language primarily used for web development. It supports prototypal inheritance, first-class functions, closures, async/await, and event-driven programming. ES6+ added classes, modules, arrow functions, destructuring, and template literals.",
    "typescript": "TypeScript is a statically typed superset of JavaScript that compiles to plain JavaScript. It adds optional static typing, interfaces, generics, union/intersection types, decorators, and advanced type inference while maintaining full JavaScript compatibility.",
    "rust": "Rust is a systems programming language focused on memory safety without garbage collection. It uses ownership, borrowing, and lifetimes to enforce safety at compile time. Key features include zero-cost abstractions, pattern matching, traits, enums, and fearless concurrency.",
    "go": "Go (Golang) is a statically typed, compiled language designed at Google for simplicity and concurrency. It features goroutines, channels, interfaces, fast compilation, a standard library with built-in HTTP server, and garbage collection. Go omits classes and inheritance in favor of composition.",
    "cpp": "C++ is a general-purpose programming language with OOP, generic programming, and low-level memory manipulation. It features templates, RAII, smart pointers, the STL (containers, algorithms, iterators), operator overloading, multiple inheritance, and manual memory management via pointers.",
    "java": "Java is a class-based, OOP language with write-once-run-anywhere via JVM bytecode. It features strong typing, automatic garbage collection, interfaces, generics, streams, lambdas, and a vast ecosystem. Java uses packages, access modifiers, and checked exceptions.",
    "csharp": "C# is a strongly typed, multi-paradigm language for .NET. It supports OOP, functional programming, async/await, LINQ, pattern matching, records, nullable reference types, and delegates/events. C# compiles to IL that runs on the CLR.",
    "ruby": "Ruby is a dynamic, object-oriented language focused on developer happiness. Everything is an object. It features blocks, procs, lambdas, mixins via modules, metaprogramding, duck typing, and the Rails web framework. Ruby uses 'do...end' and braces for blocks.",
    "swift": "Swift is Apple's statically typed language for iOS, macOS, and Linux. It features optionals, type inference, value types (structs/enums), protocols, generics, error handling with try/catch, ARC memory management, and string interpolation with backslash-paren syntax.",
    "kotlin": "Kotlin is a statically typed language that interops fully with Java. It features null safety, data classes, sealed classes, extension functions, coroutines for async, smart casts, when expressions, and is the preferred language for Android development.",
    "haskell": "Haskell is a purely functional, lazily evaluated, statically typed language. It features monads, type classes, algebraic data types, pattern matching, currying, higher-order functions, type inference, and referential transparency. Haskell uses 'let' and 'where' for bindings.",
    "elixir": "Elixir is a functional, concurrent language running on the BEAM VM. It features immutable data, pattern matching, the pipe operator, GenServer for stateful processes, supervision trees, macro metaprogramming, and fault tolerance via let-it-crash philosophy.",
    "zig": "Zig is a systems programming language designed as a modern C alternative. It features manual memory management with explicit allocators, compile-time code execution, no hidden control flow, error return values (not exceptions), and seamless C interop.",
    "nim": "Nim is a statically typed systems language that compiles to C, C++, or JavaScript. It features a clean Python-like syntax, metaprogramming via macros and templates, memory management options (GC, ARC, manual), and generic programming.",
    "lua": "Lua is a lightweight, embeddable scripting language. It uses a single table data structure for arrays, dicts, and objects. Features include first-class functions, closures, coroutines, metatables for OOP, and 1-based indexing. Lua is popular in game scripting and embedded systems.",

    # Blockchain / Web3
    "solidity": "Solidity is a statically typed smart contract language for the Ethereum Virtual Machine (EVM). It features contract-oriented programming, state variables, modifiers, events, require/assert/revert for error handling, inheritance, interfaces, and the 'payable' keyword for ETH handling. Gas optimization is critical.",
    "vyper": "Vyper is a Pythonic smart contract language for Ethereum designed for security and auditability. It deliberately omits features like modifiers, class inheritance, and inline assembly to reduce attack surface. Vyper bounds loops and enforces integer overflow checks.",
    "move": "Move is a resource-oriented programming language for the Sui and Aptos blockchains. Its key innovation is first-class resources that cannot be copied or dropped. Move uses modules, structs, abilities (copy, drop, store, key), and access control via module visibility.",
    "cairo": "Cairo is a Turing-complete language for STARK-provable programs, used in StarkNet. It features built-in cryptographic primitives, felt (field element) as the base type, implicit arguments, and the ability to generate mathematical proofs of computation.",
    "solana": "Solana is a high-performance blockchain using Proof of History (PoH) combined with PoS. Programs (smart contracts) are written in Rust and run via the BPF runtime. Key concepts include accounts, rent, cross-program invocations (CPI), and the Anchor framework for program development.",
    "cosmos": "Cosmos is an ecosystem of interoperable blockchains connected via the Inter-Blockchain Communication (IBC) protocol. It uses the Tendermint BFT consensus engine and the Cosmos SDK (Go) for building application-specific blockchains. Each chain is sovereign with its own validators.",
    "clarity": "Clarity is a decidable smart contract language for the Stacks blockchain (secured by Bitcoin). It is interpreted (not compiled), uses LISP-like syntax, and is intentionally Turing-incomplete to prevent unbounded loops. Key features include trait-based composition and post-condition checking.",
    "plutus": "Plutus is a smart contract platform on Cardano using Haskell-based Plutus Core. It separates on-chain validation scripts from off-chain code. Key concepts include the EUTXO model, validator scripts, datum, redeemer, and the Plutus Application Backend (PAB).",
    "foundry": "Foundry is a Rust-based Ethereum development toolkit including Forge (testing), Cast (interaction), Anvil (local node), and Chisel (Solidity REPL). It uses Solidity for test writing, supports fuzz testing, invariant testing, and cheatcodes via the vm pragma.",
    "hardhat": "Hardhat is a JavaScript/TypeScript-based Ethereum development environment. It features task runner, plugin system, Solidity debugging with stack traces, console.log in contracts, mainnet forking, and the Hardhat Network for local testing.",
    "web3js": "Web3.js is a JavaScript library for interacting with Ethereum nodes via JSON-RPC. It provides contract abstraction (web3.eth.Contract), event listening, transaction signing, ABI encoding/decoding, and support for providers like MetaMask and Infura.",
    "ethersjs": "Ethers.js is a JavaScript/TypeScript library for Ethereum interaction. Key classes include Provider (read-only blockchain access), Signer (transaction signing), Contract (ABI interaction), and Wallet. It supports EIP-1193 providers, ENS resolution, and typed transactions.",
    "defi": "DeFi (Decentralized Finance) refers to financial protocols built on blockchains without intermediaries. Key concepts include automated market makers (AMMs), liquidity pools, yield farming, flash loans, oracle networks, over-collateralized lending, and tokenomics. Major protocols include Uniswap, Aave, and Compound.",
    "nft": "NFTs (Non-Fungible Tokens) are unique digital assets on a blockchain. Key standards include ERC-721 (individual tokens) and ERC-1155 (multi-token). Concepts include metadata (via tokenURI), minting, royalties, on-chain vs off-chain storage, marketplaces, and royalty enforcement via EIP-2981.",

    # DevOps / Infrastructure
    "docker": "Docker is a containerization platform that packages applications with their dependencies into portable containers. Key concepts include Dockerfile (build instructions), images, containers, layers, volumes, networks, multi-stage builds, docker-compose for multi-container apps, and registry/push/pull workflows.",
    "kubernetes": "Kubernetes (K8s) is a container orchestration platform. Key concepts include Pods (smallest deployable unit), Deployments, Services, ConfigMaps, Secrets, Ingress, Namespaces, ReplicaSets, StatefulSets, DaemonSets, and kubectl for management. K8s uses declarative YAML manifests and self-healing.",
    "terraform": "Terraform is an Infrastructure as Code (IaC) tool by HashiCorp. It uses HCL (HashiCorp Configuration Language) to define resources declaratively. Key concepts include providers, resources, state files, plan/apply/destroy lifecycle, modules, variables, outputs, data sources, and the backend for state storage.",
    "ci_cd": "CI/CD (Continuous Integration / Continuous Delivery) automates building, testing, and deploying code. Key concepts include pipelines, stages, jobs, artifacts, triggers, rollbacks, blue-green deployments, canary releases, and feature flags. Tools include GitHub Actions, GitLab CI, Jenkins, and CircleCI.",
    "cloud_aws": "AWS (Amazon Web Services) is a cloud computing platform. Key services include EC2 (compute), S3 (storage), RDS (databases), Lambda (serverless), VPC (networking), IAM (identity), CloudFormation (IaC), ECS/EKS (containers), and CloudFront (CDN). AWS uses regions, availability zones, and ARNs for resource identification.",
    "cloud_gcp": "GCP (Google Cloud Platform) provides cloud services including Compute Engine (VMs), Cloud Storage, Cloud SQL, Cloud Functions, GKE (Kubernetes), BigQuery (analytics), Pub/Sub (messaging), and Cloud Build (CI/CD). GCP uses projects, zones, regions, and service accounts for access control.",
    "cloud_azure": "Microsoft Azure is a cloud platform with services including Virtual Machines, Blob Storage, Azure SQL, Azure Functions, AKS (Kubernetes), Azure DevOps, Active Directory, and App Service. Azure uses subscriptions, resource groups, regions, and RBAC for access management.",

    # Security / Cryptography
    "cryptography": "Cryptography is the science of secure communication. Key concepts include symmetric encryption (AES, DES), asymmetric encryption (RSA, ECC), hash functions (SHA-256, BLAKE2), digital signatures, MACs, key exchange (Diffie-Hellman), PKI, certificates, and zero-knowledge proofs. Security depends on key management and algorithm choice.",
    "secure_coding": "Secure coding is the practice of writing code that protects against vulnerabilities. Key concepts include input validation, output encoding, parameterized queries (SQL injection prevention), CSRF tokens, XSS prevention, principle of least privilege, defense in depth, OWASP Top 10, and secrets management.",
    "pentesting": "Penetration testing is authorized simulated attacks on systems to find vulnerabilities. Key phases include reconnaissance, scanning, exploitation, post-exploitation, and reporting. Tools include Nmap, Burp Suite, Metasploit, and Wireshark. Methodologies include black-box, white-box, and gray-box testing.",
    "network_security": "Network security protects network infrastructure from unauthorized access. Key concepts include firewalls, IDS/IPS, VPNs, TLS/SSL, network segmentation, zero-trust architecture, DDoS protection, port scanning, packet analysis, and the OSI model layers with their respective threats and controls.",
}


def load_taxonomy():
    """Load the subject taxonomy YAML file."""
    with open(TAXONOMY_PATH, "r") as f:
        return yaml.safe_load(f)


def resolve_subject(subject_input):
    """Resolve a user-provided subject name to an MMLU subject key.

    If the input is already a valid MMLU subject (in the taxonomy categories),
    return it as-is. Otherwise, try the subject_mapping.
    """
    taxonomy = load_taxonomy()

    # Collect all valid MMLU subjects
    all_subjects = set()
    for cat_subjects in taxonomy.get("categories", {}).values():
        all_subjects.update(cat_subjects)

    # If it's already a valid MMLU subject, return it
    if subject_input in all_subjects:
        return subject_input

    # Try the mapping
    mapping = taxonomy.get("subject_mapping", {})
    if subject_input in mapping:
        resolved = mapping[subject_input]
        if resolved in all_subjects:
            return resolved

    # Try replacing spaces with underscores
    candidate = subject_input.replace(" ", "_").lower()
    if candidate in all_subjects:
        return candidate

    # Try the mapping with underscores
    if candidate in mapping:
        resolved = mapping[candidate]
        if resolved in all_subjects:
            return resolved

    return None


def format_mcq_prompt(question, choices, subject=None):
    """Format a multiple-choice question prompt for the model.

    Args:
        question: The question text
        choices: List of 4 answer choices
        subject: Optional subject name for context

    Returns:
        Formatted prompt string
    """
    lines = []
    if subject:
        lines.append(f"Subject: {subject.replace('_', ' ').title()}")
        lines.append("")

    lines.append(question)
    lines.append("")

    for i, choice in enumerate(choices):
        lines.append(f"{_LETTERS[i]}) {choice}")

    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")

    return "\n".join(lines)


def is_custom_subject(subject):
    """Check if a subject is a custom (non-MMLU) subject."""
    return subject in SUBJECT_DESCRIPTIONS


def generate_custom_questions(subject, limit=25):
    """Generate MCQ questions for custom (non-MMLU) subjects from their descriptions.

    Creates 4 multiple-choice questions per subject with a correct answer key.
    Questions are deterministic (no LLM needed) — they test factual knowledge
    based on the subject description.

    Args:
        subject: Custom subject key (must be in SUBJECT_DESCRIPTIONS)
        limit: Maximum number of questions to return

    Returns:
        List of dicts with keys: question, choices, answer (int 0-3), subject
    """
    description = SUBJECT_DESCRIPTIONS.get(subject)
    if not description:
        logger.warning(f"No description found for custom subject '{subject}'")
        return []

    # Parse the description to extract key facts for question generation
    # Split into sentences for factual extraction
    sentences = [s.strip() for s in description.split(".") if s.strip()]

    questions = []

    # Generate questions based on the subject description
    # Each question targets a different aspect of the subject

    # Q1: What is this subject? (definition question)
    subject_display = subject.replace("_", " ").title()
    q1_choices = [
        description.split(".")[0].strip(),  # correct — first sentence
        "A web framework for building REST APIs",
        "A database management system",
        "A graphics rendering engine",
    ]
    questions.append({
        "question": f"Which of the following best describes {subject_display}?",
        "choices": q1_choices,
        "answer": 0,
        "subject": subject,
    })

    # Q2: Extract a key feature from the description
    # Find a sentence mentioning a key feature
    feature_sentence = None
    for s in sentences[1:]:
        if any(kw in s.lower() for kw in ["features", "uses", "supports", "includes", "key", "designed"]):
            feature_sentence = s.strip()
            break
    if not feature_sentence and len(sentences) > 1:
        feature_sentence = sentences[1].strip()

    if feature_sentence:
        q2_choices = [
            feature_sentence,  # correct
            "It only works on Windows operating systems",
            "It requires a paid license for commercial use",
            "It was deprecated in 2020",
        ]
        questions.append({
            "question": f"What is a key characteristic of {subject_display}?",
            "choices": q2_choices,
            "answer": 0,
            "subject": subject,
        })

    # Q3: Identify the domain/category
    # Determine which category this subject belongs to
    taxonomy = load_taxonomy()
    category_label = "technology"
    for cat, subjects_list in taxonomy.get("categories", {}).items():
        if subject in subjects_list:
            category_label = cat.replace("_", " ").lower()
            break

    q3_choices = [
        f"It belongs to the {category_label} domain",
        "It is a biological science",
        "It is a form of classical music",
        "It is a type of medieval architecture",
    ]
    questions.append({
        "question": f"Which domain does {subject_display} belong to?",
        "choices": q3_choices,
        "answer": 0,
        "subject": subject,
    })

    # Q4: Extract another distinguishing feature
    # Find a sentence with technical keywords
    tech_sentence = None
    for s in sentences[1:]:
        s_lower = s.lower()
        if any(kw in s_lower for kw in ["type", "memory", "compile", "runtime", "syntax", "concurr",
                                          "blockchain", "smart contract", "contract", "container", "encrypt",
                                          "protocol", "framework", "library", "agent", "statically",
                                          "dynamically", "object", "functional", "oriented", "service"]):
            tech_sentence = s.strip()
            break

    if tech_sentence:
        q4_choices = [
            tech_sentence,  # correct
            "It has no support for any data structures",
            "It cannot run on modern hardware",
            "It uses punch cards for input",
        ]
        questions.append({
            "question": f"Which statement about {subject_display} is correct?",
            "choices": q4_choices,
            "answer": 0,
            "subject": subject,
        })

    logger.info(f"Generated {len(questions)} custom questions for subject '{subject}'")
    return questions[:limit]


def load_mmlu_questions(subject, limit=25, split="dev"):
    """Load MMLU questions for a subject from HuggingFace datasets.

    Args:
        subject: MMLU subject key (e.g., 'high_school_mathematics')
        limit: Maximum number of questions to return
        split: Dataset split to use ('dev', 'validation', 'test', 'auxiliary_train')

    Returns:
        List of dicts with keys: question, choices, answer (int 0-3), subject
    """
    from datasets import load_dataset

    logger.info(f"Loading MMLU dataset (config='all', split='{split}')...")

    try:
        ds = load_dataset("cais/mmlu", "all", split=split)
    except Exception as e:
        logger.warning(f"Failed to load split '{split}': {e}")
        # Fall back to 'dev' split which is small and always available
        if split != "dev":
            logger.info("Falling back to 'dev' split...")
            ds = load_dataset("cais/mmlu", "all", split="dev")
        else:
            raise

    # Filter by subject
    subject_data = ds.filter(lambda x: x["subject"] == subject)

    questions = []
    for item in subject_data:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "subject": item["subject"],
        })
        if len(questions) >= limit:
            break

    logger.info(f"Loaded {len(questions)} questions for subject '{subject}'")

    if len(questions) < limit:
        # If we don't have enough in this split, try the test split
        logger.info(f"Only {len(questions)} questions in '{split}' split. Trying 'test' split...")
        try:
            ds_test = load_dataset("cais/mmlu", "all", split="test")
            subject_test = ds_test.filter(lambda x: x["subject"] == subject)

            for item in subject_test:
                questions.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": item["subject"],
                })
                if len(questions) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Failed to load test split: {e}")

    logger.info(f"Total questions loaded: {len(questions)}")

    return questions[:limit]


# ─── Tier 2: Depth Probing ──────────────────────────────────────────

def generate_tier2_followups(question, model_answer, subject):
    """Generate Tier 2 (depth) follow-up questions.

    Tier 2 drills deeper into the model's answer to test depth of
    understanding. Follow-ups ask the model to explain, justify,
    or elaborate on its Tier 1 answer.

    Args:
        question: The original Tier 1 question text
        model_answer: The model's Tier 1 answer text
        subject: The subject area (e.g., 'high_school_mathematics')

    Returns:
        list[str]: Follow-up questions for Tier 2 probing
    """
    subject_display = subject.replace("_", " ").title()

    followups = [
        f"Regarding your answer to the previous question about {subject_display}, "
        f"can you explain the reasoning behind your answer? Walk through your thought process step by step.",

        f"You answered: \"{model_answer[:200]}\". "
        f"Why is that the correct answer? What principles or concepts support this?",

        f"If someone disagreed with your answer, what counterargument might they make? "
        f"How would you defend your answer against that objection?",

        f"What are the key assumptions underlying your answer to the question about {subject_display}? "
        f"Are there cases where those assumptions might not hold?",

        f"Can you provide a specific example or illustration that demonstrates why your answer is correct?",
    ]

    return followups


# ─── Tier 3: Edge Cases & Adversarial ──────────────────────────────

# Subject-specific common misconceptions for adversarial questions
_MISCONCEPTIONS = {
    "high_school_mathematics": [
        "A common misconception is that (a+b)^2 = a^2 + b^2. Explain why this is incorrect and provide the correct expansion.",
        "Some students believe that the derivative of a product is the product of derivatives. Is this correct? Explain why or why not.",
        "Is 0.999... equal to 1? Many people believe they are different. Explain the correct mathematical reasoning.",
    ],
    "high_school_physics": [
        "A common misconception is that heavier objects fall faster than lighter ones in a vacuum. Explain why this is incorrect.",
        "Some people believe that a force is always needed to keep an object moving. Is this true? Explain using Newton's laws.",
        "Is it true that energy is always conserved in every process? What about in inelastic collisions?",
    ],
    "high_school_chemistry": [
        "A common misconception is that atoms are the smallest particles that exist. Is this accurate? Explain.",
        "Some students believe that chemical bonds store energy. Explain why this is incorrect and what actually happens.",
        "Is it true that all reactions release energy? Distinguish between exothermic and endothermic reactions.",
    ],
    "high_school_biology": [
        "A common misconception is that evolution is a random process. Explain the role of natural selection.",
        "Some people believe that DNA directly produces proteins. What step is missing? Explain.",
        "Is it true that all bacteria are harmful? Explain the role of beneficial bacteria.",
    ],
    "high_school_us_history": [
        "A common misconception is that the Civil War was primarily about states' rights. Explain the central role of slavery.",
        "Some believe the American Revolution was universally popular. Explain the role of loyalists.",
        "Is it true that the Great Depression was caused solely by the stock market crash? Explain other contributing factors.",
    ],
    "high_school_world_history": [
        "A common misconception is that the Dark Ages were a period of no progress. Explain what was actually happening.",
        "Some believe the fall of Rome was sudden. Explain the gradual nature of the collapse.",
        "Is it accurate to say the Renaissance was a complete break from medieval thinking? Explain the continuity.",
    ],
}

# Generic edge-case questions that work for any subject
_GENERIC_EDGE_CASES = [
    "What is a common misconception about this topic that many people hold? Explain why it is wrong.",
    "What is an edge case or exception to the general rule you just described? When does the typical answer not apply?",
    "If the question conditions were slightly different (e.g., changed assumptions), how would the answer change?",
    "What would happen in the extreme case or boundary condition of this problem?",
    "What is a tricky or deceptive version of this question that might fool a student?",
]


def generate_tier3_edge_cases(subject):
    """Generate Tier 3 (edge cases) adversarial questions.

    Tier 3 tests the model's ability to handle trick questions,
    common misconceptions, and boundary conditions.

    Args:
        subject: The subject area (e.g., 'high_school_mathematics')

    Returns:
        list[str]: Adversarial/edge-case questions for Tier 3 probing
    """
    # Try subject-specific misconceptions first
    specific = _MISCONCEPTIONS.get(subject, [])
    result = list(specific)

    # Add generic edge cases
    result.extend(_GENERIC_EDGE_CASES)

    return result
