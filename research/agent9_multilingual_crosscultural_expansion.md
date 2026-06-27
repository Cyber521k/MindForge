# MindForge Domain Expansion: Multilingual & Cross-Cultural Knowledge

## Agent 9 — Domain Expansion Researcher

## Research Context

MindForge's current taxonomy (taxonomy/subjects.yaml) has 10 domains with ~100+ subjects, but is entirely English-centric:
- STEM, Humanities, Social Science, Professional, Other (from MMLU — English-language benchmark)
- Programming_Languages, Agent_Frameworks, Blockchain_Web3, DevOps_Infrastructure, Security_Cryptography (custom additions)

Frontier models like Qwen 3.5 (201 languages), Gemini 3.1 (multimodal), GPT-5, Claude, and Llama 4 claim multilingual competence. MindForge has no way to test this. This file proposes new domains and subjects to fill that gap.

## Current Domain Count: 10
## Proposed New Domains: 5
## Total New Subjects Identified: 52

---

## New Domain 1: Multilingual_Linguistics

Tests whether models understand language structure, typology, and translation across the world's languages — not just English.

| # | Subject Key | Description |
|---|------------|-------------|
| 1 | `linguistic_typology` | Classification of languages by structural features (isolating, agglutinative, fusional, polysynthetic). Morphological and syntactic typology, word order patterns (SOV, SVO, VSO), grammatical hierarchies. |
| 2 | `comparative_linguistics` | Historical/comparative method, cognate identification, sound correspondence, reconstruction of proto-languages (Proto-Indo-European, Proto-Sino-Tibetan, Proto-Bantu). Language families and genetic classification. |
| 3 | `translation_theory` | Equivalence theories (Nida, Catford), dynamic vs formal equivalence, untranslatability, cultural translation, localization, machine translation evaluation (BLEU, chrF, COMET). |
| 4 | `sociolinguistics` | Language variation, dialects, registers, code-switching, diglossia, language planning, language death and revitalization, prestige varieties, language and identity. |
| 5 | `phonology_crossling` | Phoneme inventories across languages, tone systems, syllable structure, prosody, stress and intonation patterns, phonological processes (assimilation, harmony, lenition). IPA transcription. |
| 6 | `morphology_crossling` | Inflectional and derivational morphology across language families, agglutination, reduplication, infixation, suppletion, templatic morphology (Semitic root-and-pattern). |
| 7 | `syntax_crossling` | Syntactic theory across languages, dependency vs constituency grammars, grammatical relations, case systems, alignment (nominative-accusative, ergative-absolutive, split-S), relative clauses cross-linguistically. |
| 8 | `semantics_crossling` | Lexical semantics across languages, color term systems, kinship terminologies, spatial reference frames (absolute vs relative), evidentiality, mirativity, modality and mood systems. |
| 9 | `writing_systems` | Orthography types (alphabetic, syllabic, logographic, abjad, abugida), scripts (Latin, Cyrillic, Arabic, Devanagari, Hanzi, Hangul, Hiragana/Katakana, Ethiopic, Cherokee), orthographic depth, transliteration. |
| 10 | `low_resource_languages` | Languages with limited NLP resources, endangered languages, underrepresented language families (Niger-Congo, Austronesian, Tupian, Na-Dene, Pama-Nyungan), resource gap in multilingual AI. |
| 11 | `pragmatics_crossling` | Speech acts across cultures, politeness theories (Brown & Levinson, face), implicature, deixis, discourse markers cross-linguistically, conversational norms. |
| 12 | `computational_linguistics` | Multilingual NLP, cross-lingual transfer learning, multilingual embeddings (mBERT, XLM-R), tokenization for non-Latin scripts, low-resource MT, multilingual LLM architecture. |

## New Domain 2: Cross_Cultural_Philosophy

Tests knowledge of philosophical traditions beyond the Western canon. The current "philosophy" subject is entirely Western.

| # | Subject Key | Description |
|---|------------|-------------|
| 13 | `confucianism` | Confucian philosophy: Ren (humaneness), Li (ritual), Xiao (filial piety), Junzi (noble person), the Five Relationships, Mencius vs Xunzi on human nature, Neo-Confucianism (Zhu Xi, Wang Yangming). |
| 14 | `daoism` | Daoist philosophy: Dao (the Way), Wuwei (non-action), Yin-Yang, Zhuangzi's relativism, Laozi's Daodejing, religious Daoism vs philosophical Daoism, wuxing (five phases). |
| 15 | `buddhist_philosophy` | Four Noble Truths, Eightfold Path, Madhyamaka (Nagarjuna, emptiness/sunyata), Yogacara (mind-only), Zen/Chan, Theravada vs Mahayana, dependent origination, no-self (anatta). |
| 16 | `indian_philosophy` | Hindu philosophical schools (Nyaya, Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta), Advaita vs Dvaita, Brahman/Atman, karma and rebirth, epistemology (pramanas). |
| 17 | `islamic_philosophy` | Falsafa vs Kalam, Avicenna (Ibn Sina), Averroes (Ibn Rushd), Al-Ghazali's critique, Mu'tazilite vs Ash'arite theology, falsafa-Aristotelian synthesis, Sufi metaphysics (Ibn Arabi). |
| 18 | `african_philosophy` | Ubuntu philosophy (humanness, "I am because we are"), ethnophilosophy (Placide Tempels), professional vs philosophic sagacity (H. Odera Oruka), Negritude, postcolonial African thought. |
| 19 | `japanese_philosophy` | Zen Buddhist philosophy, Kyoto School (Nishida Kitaro, Tanabe, Nishitani), bushido ethics, Shinto cosmology, wabi-sabi aesthetics, nothingness and absolute nothing. |
| 20 | `latin_american_philosophy` | Liberation philosophy (Dussel), mestizaje identity, indigenous thought (Aztec/Nahua, Maya, Quechua philosophy), borderlands theory (Anzaldua), decolonial philosophy (Mignolo, Quijano). |
| 21 | `comparative_philosophy` | Cross-tradition philosophical comparison, methodological issues in comparative philosophy, East-West dialogue, universalism vs particularism, incommensurability of traditions. |
| 22 | `indigenous_philosophy` | Native American philosophy (relational ontology, land-based ethics), Australian Aboriginal dreaming (Tjukurpa), Maori whakapapa and kaitiakitanga, Andean sumaq kawsay (buen vivir). |

## New Domain 3: World_History_Regional

The current "high_school_world_history" is Western-centric. These subjects test region-specific historical knowledge.

| # | Subject Key | Description |
|---|------------|-------------|
| 23 | `east_asian_history` | Chinese dynastic history (Xia to Qing), Korean Three Kingdoms and Joseon, Japanese Heian/Edo/Meiji periods, Mongol Empire, tributary system, Sino-Japanese wars, Cultural Revolution. |
| 24 | `south_asian_history` | Indus Valley Civilization, Mauryan and Gupta Empires, Delhi Sultanate, Mughal Empire, British Raj, Indian independence (Gandhi, Nehru), Partition, post-colonial India/Pakistan/Bangladesh. |
| 25 | `southeast_asian_history` | Srivijaya and Majapahit empires, Khmer Empire/Angkor, Ayutthaya/Siam, Vietnam under Chinese rule and independence, colonial era (Dutch East Indies, French Indochina), ASEAN formation. |
| 26 | `middle_eastern_history` | Mesopotamian civilizations, Persian empires (Achaemenid, Sassanid, Safavid), Islamic conquests and caliphates (Rashidun, Umayyad, Abbasid), Ottoman Empire, modern Middle East, Arab Spring. |
| 27 | `african_history` | Ancient Egypt and Nubia, trans-Saharan trade, West African empires (Ghana, Mali, Songhai), Great Zimbabwe, Atlantic slave trade, colonial scramble for Africa, decolonization, post-independence conflicts. |
| 28 | `latin_american_history` | Pre-Columbian civilizations (Maya, Aztec, Inca), Spanish/Portuguese conquest, colonial period, independence movements (Bolivar, San Martin), 20th century revolutions (Cuba, Nicaragua), dictatorship to democracy. |
| 29 | `central_asian_history` | Silk Road, Turkic and Mongol khanates, Timurid Empire, Russian conquest, Soviet era, post-Soviet independence, Great Game, nomadic empires and sedentary relations. |
| 30 | `pacific_islander_history` | Polynesian/Micronesian/Melanesian migrations, Hawaiian Kingdom, Maori settlement of Aotearoa, colonial era, nuclear testing in Pacific, decolonization movements, ANZUS and Pacific geopolitics. |

## New Domain 4: Cultural_Knowledge

Tests knowledge of cultural products, practices, and traditions that are language-embedded and culturally specific.

| # | Subject Key | Description |
|---|------------|-------------|
| 31 | `world_mythology` | Comparative mythology: Greek, Norse, Egyptian, Mesopotamian, Hindu, Chinese, Yoruba, Aztec, Polynesian myth cycles. Hero's journey (Campbell), structural mythology (Levi-Strauss), etiological myths. |
| 32 | `folklore_studies` | Folklore genres (tales, legends, myths, jokes, riddles), performance theory, folktale types (Aarne-Thompson-Uther index), folklore and identity, oral tradition vs literary tradition. |
| 33 | `proverbs_idioms` | Proverb collections across cultures, paremiology, proverb semantics, cultural metaphors in idioms, untranslatable idioms, proverb functions in discourse, cross-cultural proverb equivalence. |
| 34 | `world_literature` | Literary traditions beyond the Western canon: The Tale of Genji, Dream of the Red Chamber, One Thousand and One Nights, Mahabharata/Ramayana, Persian poetry (Rumi, Hafez), Latin American magical realism. |
| 35 | `comparative_literature` | Cross-cultural literary analysis, influence and reception studies, world literature theory (Moretti, Casanova, Damrosch), translation and literary circulation, postcolonial literature. |
| 36 | `ethnomusicology` | Musical traditions worldwide: Indian classical (raga, tala), Arabic maqam, West African drumming, gamelan, Japanese gagaku, Andean music, blues and African diasporic music. Organology, musical transcription. |
| 37 | `culinary_traditions` | Food as cultural knowledge: regional cuisines, culinary techniques, food symbolism and ritual, fermentation/preservation traditions, spice trade influence on cuisines, food taboos and religious dietary laws. |
| 38 | `traditional_medicine` | Ayurveda, Traditional Chinese Medicine (TCM, acupuncture, herbal formulas), Unani-Tibb, African traditional healing, shamanic practices, ethnomedicine, WHO traditional medicine strategy, evidence debates. |
| 39 | `ethnobotany` | Plant knowledge across cultures: medicinal plants, agricultural origins and crop domestication, sacred plants, dye plants, traditional ecological knowledge of flora, bioprospecting and indigenous rights. |
| 40 | `cultural_anthropology` | Kinship systems (descent, marriage, alliance), rites of passage, cultural relativism, ethnographic methods, magic/witchcraft/religion, economic anthropology (gift economies, potlatch). |
| 41 | `cross_cultural_communication` | High-context vs low-context cultures (Hall), Hofstede's cultural dimensions, intercultural competence, cultural shock and adaptation, nonverbal communication across cultures, intercultural conflict. |
| 42 | `world_religions_detailed` | Beyond surface knowledge: Hindu sects (Vaishnavism, Shaivism, Shaktism), Buddhist schools (Theravada, Mahayana, Vajrayana), Islamic jurisprudence (Hanafi, Maliki, Shafi'i, Hanbali, Ja'fari), Sikhism, Jainism, Shinto, Bahai, Zoroastrianism, Cao Dai. |

## New Domain 5: Comparative_Law_Governance

Tests knowledge of legal and governance systems worldwide — not just Anglo-American common law.

| # | Subject Key | Description |
|---|------------|-------------|
| 43 | `civil_law_systems` | Civil law tradition: Roman law roots, Code Napoleon, German BGB, structure of civil codes, role of judges vs legislators, inquisitorial procedure, civil law countries (France, Germany, Japan, Brazil, Egypt). |
| 44 | `islamic_law` | Sharia: sources (Quran, Sunnah, ijma, qiyas), schools of jurisprudence (madhahib), fiqh categories, maqasid al-sharia, family law, Islamic finance, modern application in Muslim-majority countries. |
| 45 | `customary_law` | Indigenous and traditional legal systems: African customary law, Aboriginal law, Native American tribal law, Melanesian customary land tenure, dispute resolution mechanisms, oral legal traditions, state-customary interaction. |
| 46 | `comparative_constitutional_law` | Constitutional systems worldwide: parliamentary vs presidential, federal vs unitary, judicial review models, constitutional courts, federalism designs, emergency powers, comparative bill of rights. |
| 47 | `international_law_comparative` | Public international law, treaty systems, ICJ, international humanitarian law (Geneva Conventions), international criminal law (ICC), human rights treaties, state sovereignty, UN system. |
| 48 | `east_asian_legal_systems` | Chinese socialist legal system, Japanese postwar constitution, Korean law, Taiwan legal system, legal transplants and adaptation, Confucian influence on law, administrative law in East Asia. |
| 49 | `african_legal_systems` | Plural legal systems (civil/customary/religious), post-colonial legal transplants, African Union legal framework, customary courts, South African constitutional court, land law and reform. |
| 50 | `latin_american_legal_systems` | Iberian civil law heritage, constitutionalism in Latin America, indigenous rights law, transitional justice, Inter-American Court of Human Rights, legal pluralism, Bolivian/Mexican/Ecuadorian plurinational law. |
| 51 | `indigenous_governance` | Traditional governance structures: Iroquois confederacy, Maori iwi/hapu governance, Navajo Nation, Sami parliament, self-determination and sovereignty claims, UN Declaration on the Rights of Indigenous Peoples (UNDRIP). |
| 52 | `comparative_political_systems` | Political systems worldwide: Westminster, consociationalism, single-party systems, theocracy, military governments, hybrid regimes, electoral systems comparison, democratization theory. |

---

## Summary

| Metric | Count |
|--------|-------|
| New domains | 5 |
| New subjects (Multilingual_Linguistics) | 12 |
| New subjects (Cross_Cultural_Philosophy) | 10 |
| New subjects (World_History_Regional) | 8 |
| New subjects (Cultural_Knowledge) | 12 |
| New subjects (Comparative_Law_Governance) | 10 |
| **Total new subjects** | **52** |

## Rationale per Domain

**Multilingual_Linguistics** — Frontier models claim 100+ language support. Testing linguistic typology, phonology, and morphology across families validates whether models truly understand language structure or just pattern-match English-centric surface forms. Low-resource language coverage is the #1 gap in multilingual LLM evaluation (INCLUDE benchmark, MuBench).

**Cross_Cultural_Philosophy** — Current "philosophy" subject is Western-only. Models claiming general intelligence should know Confucian ethics, Buddhist epistemology, Islamic falsafa, Ubuntu, and Kyoto School thought. These traditions have millions of adherents and distinct philosophical frameworks. The culturALL benchmark showed <45% accuracy on cultural competence.

**World_History_Regional** — "high_school_world_history" in MMLU is Eurocentric. Region-specific history tests whether models know East Asian dynastic cycles, pre-colonial African empires, Southeast Asian maritime trade networks, and Pacific Islander migrations — knowledge native speakers of those languages would expect.

**Cultural_Knowledge** — Proverbs, idioms, mythology, folklore, and ethnomusicology are deeply language-embedded. A model claiming fluency in Chinese should know the Monkey King; a model claiming Arabic fluency should know Rumi and the maqam tradition. These subjects bridge language competence and cultural intelligence. Scale AI's MultiNRC benchmark explicitly tests this kind of natively-built, culturally-aware knowledge.

**Comparative_Law_Governance** — Legal systems are fundamentally cultural products. The current taxonomy has "professional_law" (US-centric) and "international_law" (Western treaty law). Missing: Islamic jurisprudence, customary/indigenous law, civil law traditions, East Asian socialist law. Models used in global contexts need this knowledge.

## Integration Notes

To integrate these into MindForge:

1. Add the 5 new domain categories to `taxonomy/subjects.yaml` under `categories:`
2. Add subject aliases to `subject_mapping:` (e.g., `typology: linguistic_typology`, `daoism: daoism`, `sharia: islamic_law`)
3. Add entries to `SUBJECT_DESCRIPTIONS` in `mindforge/probe/question_gen.py` for each of the 52 subjects
4. Update `test_domain_expansion.py` to expect 15 domains instead of 10
5. The custom question generator (`generate_custom_questions`) will auto-generate MCQs from the descriptions

## File Path
/Users/cyber521k/MindForge/research/agent9_multilingual_crosscultural_expansion.md
