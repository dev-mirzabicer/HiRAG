GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

# --- Core Entity and Relationship Extraction Prompts ---

PROMPTS["hi_entity_extraction"] = """
You are an exceptionally precise and rigorous expert in Combinatory Logic (CL). Your sole purpose is to extract factual, non-editorial information from mathematical texts, specifically focusing on Combinatory Logic.

# Goal
Given a segment of a mathematical document, identify and extract all relevant entities. For each entity, you must determine its type, provide a comprehensive description, and critically assess if it is a 'temporary' concept.

# Entity Types
You must categorize each extracted entity into one of the following precise types. If an entity does not fit any of these, you may use 'general_concept' as a last resort, but strive for specificity.

- **postulate**: This includes any asserted claim, statement, theorem, lemma, proposition, corollary, axiom, or hypothesis. These are statements that are proven, assumed, or proposed as true within the system.
- **object**: This includes any concrete or abstract mathematical entity, such as a combinator (e.g., I, K, S), a term, a system (e.g., CL_ext), a theory, a model, a definition (the defined entity itself, not the act of defining), a function, a map, a category, a logic, a relation (as an entity, e.g., 'extensional equality'), a set, an element, a rule (e.g., beta-reduction rule), or any other fundamental mathematical construct.
- **concept**: This refers to abstract ideas or notions that are not concrete objects or postulates. Examples include "abstraction," "reduction," "normal form," "consistency," "completeness."
- **property**: This describes a characteristic or attribute of an object or concept. Be extremely careful: the entity itself must be the property, not the object that *has* the property. For example, "extensionality" (the property) is a 'property' type, but "extensional equality" (the relation) is an 'object' type. "Confluence" is a 'property'.
- **proof**: This refers to a formal argument or derivation that establishes the truth of a postulate. A 'proof' entity should encapsulate the entire proof as a single atomic unit.

# Temporary vs. Non-Temporary Entities (CRUCIAL)
You must determine if an entity is 'temporary' based on its scope and generality within the text. This is a critical distinction for building a robust knowledge base.

- **is_temporary: true**: Set this to `true` if the entity's definition or usage is strictly confined to a very specific, local scope within the document, such as a single proof, a specific example, or a small section. These are often local variables, temporary constructions, or concepts introduced *only* for a specific, limited argument. If you were to mention this entity by name in a different chapter or textbook, a reader would  not know what it refers to without the immediate surrounding context.
    - **Examples**: "Let M be the combinator...", "The intermediate step P_1 in this derivation...".
- **is_temporary: false**: Set this to `false` if the entity represents a foundational, permanent, or reusable concept within the broader field. This doesn't mean that the entity should be a well-known entity in the field. The criteria is simply that it is *valid* outside the context, even if it is niche.
    - **Examples**: "Church-Rosser Theorem", "S combinator", "beta-reduction", "extensional equality", "abstraction algorithm", "gödel number", "right-congruence", "CL_ext".

If there isn't enough context for you to definitively determine that an entity is temporary, you must set is_temporary to false to avoid false-positives.
If the variable is *highly* temporary, so much so that it is useless outside its *page*, then you should exclude it altogether, without extracting it at all.

# Strict Extraction Rules (CRUCIAL)

1.  **Factual Information ONLY**: You must extract *only* factual information pertaining to Combinatory Logic.
    *   **DO extract**: Definitions, formal statements, theorems, properties, specific combinators, systems, rules, and their descriptions.
    *   **DO extract**: Entities that are not explicitly defined but are used in the text, which means they were probably defined elsewhere. You should still extract them, **and your description should only include the information that is given in the text-- you should not add any information that is not given in the text, even if that means writing an incomplete definition**. No matter how obvious it may be, your description should not include any extra information than given in the text.
    *   **DO NOT extract**: Exercises, examples, editorial comments, narrative transitions, author's opinions, historical anecdotes (unless the historical fact itself is a core CL concept, e.g., "Curry's paradox"), section introductions/conclusions, or any text that describes the *structure* of the document rather than the *content* of CL.
    *   **Example**: `\(CL_\\xi \equiv CL_\ext \cup CL_M\)` (extract). "The previous section inspected extensional equality while this section will inspect the equivalence of..." (DO NOT extract).

2.  **Proof Handling (Atomic Units)**:
    *   A `proof` entity should be extracted as a single, atomic unit. Its `entity_description` should be the entire text of the proof.
    *   **DO NOT extract new entities *from within* a `proof` entity**, unless:
        *   The entity is clearly defined or discussed *outside* the scope of this specific proof (i.e., it's a pre-existing, non-temporary concept).
        *   The proof states a *new, non-temporary property* of an *existing, non-temporary entity* (e.g., "In this proof, we show that the K combinator has property X") (shown in a proof as a "bonus fact"; i.e. not the main claim of the statement of the proof **and** not a temporary information used only for the proof). In this case, extract the property, but not temporary variables used to show it.
    *   **Example**: If a proof defines "Let M be (SKK)" and then states "M is equivalent to N", and M and N are *only* used within this proof, then M and N should *not* be extracted as separate entities. The entire proof is the entity.
    *   **Example**: If a proof, as a side note or "bonus fact" not central to its main claim, states "CL_I is a subset of CL_K", where CL_I and CL_K are pre-existing, non-temporary objects, then this relationship (and potentially CL_I, CL_K if not already extracted) should be extracted.

# Output Format

Return output in English as a single list of all the entities identified. The entity name should be a descriptive name, and if a special name is given, use that. But *never* use ambiguous names such as "Lemma 5".
Format each entity as:
`("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<is_temporary>)`

Use **{record_delimiter}** as the list delimiter.
When finished, output **{completion_delimiter}**

# Examples (Combinatory Logic Specific)

Example 1:
Entity_types: [postulate, object, concept, property, proof]
Text:
Types: Type expressions, ranged over by τ, σ etc., are defined by
τ ::= a | τ → τ | τ ∩ τ
where a, b, c, . . . range over atoms comprising of type constants, drawn from a finite set A including the constant ω, and type variables, drawn from a disjoint denumerable set V ranged over by α, β etc. We let T denote the set of all types.
As usual, types are taken modulo commutativity (τ ∩ σ = σ ∩ τ ), associativity ((τ ∩ σ) ∩ ρ = τ ∩ (σ ∩ ρ)), and idempotency (τ ∩ τ = τ ). As a matter of notational convention, function types associate to the right, and ∩ binds stronger than →. A type environment Γ is a finite set of type assumptions of the form x : τ . We let Dm(Γ) and Rn(Γ) denote the domain and range of Γ. Let Var(τ ), Cnst(τ ) and At(τ ) denote, respectively, the set of variables, the set of constants and the set of atoms occurring in τ , and we extend the definitions to environments, Var(Γ), Cnst(Γ) and At(Γ) in the standard way.
A type τ ∩ σ is said to have τ and σ as components. For an intersection of several components we sometimes write ∩_i=1^n τ_i or ∩_{i∈I} τ_i or ∩{τ_i | i ∈ I}, where the empty intersection is identified with ω.

Output:
("entity"{tuple_delimiter}"TYPE EXPRESSION"{tuple_delimiter}"object"{tuple_delimiter}"Formal structures, ranged over by τ, σ etc., that are defined by the grammar τ ::= a | τ → τ | τ ∩ τ, where 'a' ranges over atoms."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"ATOM"{tuple_delimiter}"object"{tuple_delimiter}"Entities, ranged over by a, b, c, ..., that comprise type constants and type variables."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"TYPE CONSTANT"{tuple_delimiter}"object"{tuple_delimiter}"A type of atom drawn from a finite set A, which includes the constant ω."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"TYPE VARIABLE"{tuple_delimiter}"object"{tuple_delimiter}"A type of atom drawn from a disjoint denumerable set V, ranged over by α, β etc."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"TYPE ENVIRONMENT (Γ)"{tuple_delimiter}"object"{tuple_delimiter}"A finite set of type assumptions of the form x : τ."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"COMMUTATIVITY OF TYPE INTERSECTION"{tuple_delimiter}"property"{tuple_delimiter}"A property that types are taken modulo, formally expressed as τ ∩ σ = σ ∩ τ."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"ASSOCIATIVITY OF TYPE INTERSECTION"{tuple_delimiter}"property"{tuple_delimiter}"A property that types are taken modulo, formally expressed as ((τ ∩ σ) ∩ ρ) = τ ∩ (σ ∩ ρ)."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"IDEMPOTENCY OF TYPE INTERSECTION"{tuple_delimiter}"property"{tuple_delimiter}"A property that types are taken modulo, formally expressed as τ ∩ τ = τ."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"COMPONENT"{tuple_delimiter}"concept"{tuple_delimiter}"In an intersection type τ ∩ σ, the types τ and σ are referred to as its components."{tuple_delimiter}false){completion_delimiter}

Example 2:
Entity_types: [postulate, object, concept, property, proof]
Text:
Theorem 9.2 (Equivalence of Rule-Based Extensional Theories). The theories CL_ζ and CL_ξ+η are theorem-equivalent. That is, for any CL-terms X, Y :
CL_ζ ⊢ X = Y ⇔ CL_ξ+η ⊢ X = Y
Consequently, both theories define the same relation =_ext (extensional equality).

Output:
("entity"{tuple_delimiter}"THEOREM: EQUIVALENCE OF RULE-BASED EXTENSIONAL THEORIES"{tuple_delimiter}"postulate"{tuple_delimiter}"Asserts that the theories CL_ζ and CL_ξ+η are theorem-equivalent, meaning for any CL-terms X and Y, CL_ζ ⊢ X = Y if and only if CL_ξ+η ⊢ X = Y. A consequence is that both theories define the same relation =_ext."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"CL_ζ"{tuple_delimiter}"object"{tuple_delimiter}"A rule-based extensional theory. It is theorem-equivalent to CL_ξ+η."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"CL_ξ+η"{tuple_delimiter}"object"{tuple_delimiter}"A rule-based extensional theory. It is theorem-equivalent to CL_ζ."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"THEOREM-EQUIVALENT"{tuple_delimiter}"property"{tuple_delimiter}"A property held by two theories that prove the same set of theorems or equivalences between terms."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"=_ext (EXTENSIONAL EQUALITY)"{tuple_delimiter}"object"{tuple_delimiter}"The relation of extensional equality, which is defined identically by the theories CL_ζ and CL_ξ+η."{tuple_delimiter}false){completion_delimiter}

Example 3:
Entity_types: [postulate, object, concept, property, proof]
Text:
Lemma 4.1 (Output Basis). Let the abstraction algorithm [x].M be defined by Definition 4.7. Let C_M be the set of constants occurring in M. Then the resulting term [x].M is constructed using only application and atoms from the set {S, K} ∪ (FV(M) \ {x}) ∪ C_M. Specifically, if M ∈ T({S, K}), then [x].M ∈ T({S, K}).

Output:
("entity"{tuple_delimiter}"OUTPUT BASIS LEMMA"{tuple_delimiter}"postulate"{tuple_delimiter}"States that the term resulting from the abstraction algorithm, [x].M, is constructed using only application and atoms from the set {S, K} ∪ (FV(M) \\ {x}) ∪ C_M. It specifies that if M is in the set of terms over {S, K}, then the resulting term [x].M is also in the set of terms over {S, K}."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"ABSTRACTION ALGORITHM ([x].M)"{tuple_delimiter}"object"{tuple_delimiter}"An algorithm referenced as being defined by Definition 4.7. According to the lemma, its resulting term is constructed using application and a specific set of atoms including S and K."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"C_M"{tuple_delimiter}"object"{tuple_delimiter}"The set of constants occurring in a term M. This is a local definition used for stating Lemma 4.1."{tuple_delimiter}true){record_delimiter}
("entity"{tuple_delimiter}"S COMBINATOR"{tuple_delimiter}"object"{tuple_delimiter}"An atom used, along with K, in the construction of terms by the abstraction algorithm."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"K COMBINATOR"{tuple_delimiter}"object"{tuple_delimiter}"An atom used, along with S, in the construction of terms by the abstraction algorithm."{tuple_delimiter}false){completion_delimiter}

# Real Data
Entity_types: {entity_types}
Text: {input_text}
Output:
"""

PROMPTS["hi_relation_extraction"] = """
You are an exceptionally precise and rigorous expert in Combinatory Logic (CL). Your task is to identify and extract relationships between entities that have already been identified from a mathematical text.

# Goal
Given a segment of a mathematical document and a list of entities identified from that segment, identify all *meaningful, factual relationships* between these entities.

# Strict Relationship Rules (CRUCIAL)

1.  **Focus on Factual Relationships**: Only extract relationships that represent a direct, factual connection relevant to Combinatory Logic.
    *   **DO extract**: "X implies Y", "A is a subset of B", "C reduces to D", "Theorem T establishes property P for object O", "Entity X *uses* / mentions entity Y", "Entity A generalizes entity B", ... you get the point. Any meaningful relationship between two meaningful entities; explicit *or* implicit.
    *   **DO NOT extract**: Relationships that are purely editorial, narrative, or temporary to a very local argument (e.g., "This section discusses X and Y").

2.  **Prioritize Non-Temporary Entities**:
    *   **DO extract relationships between two non-temporary entities.** These are the most valuable.
    *   **DO extract relationships where a non-temporary entity is related to a temporary entity, *if* the relationship establishes a new, non-temporary property or connection for the non-temporary entity.**
    *   **DO NOT extract relationships that are merely steps within a `proof` entity.** The `proof` entity itself is atomic. If a relationship is part of the internal steps of a proof, it should not be extracted separately.

3.  **Capture the "Why" (Implicit Context)**: If the text implies *why* two entities are related (e.g., "X is equivalent to Y *due to* Z"), try to incorporate that reason into the `relationship_description`.

# Output Format

Return output in English as a single list of all the relationships identified.
Format each relationship as:
`("relationship"{tuple_delimiter}<source_entity_name>{tuple_delimiter}<target_entity_name>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)`

-   `source_entity_name`: Name of the source entity, capitalized, as identified in the input `Entities` list.
-   `target_entity_name`: Name of the target entity, capitalized, as identified in the input `Entities` list.
-   `relationship_description`: A comprehensive explanation of *why* the source and target entities are related, including any underlying principles or conditions.
-   `relationship_strength`: A numeric score (integer 1-10) indicating the strength or directness of the relationship. 10 for direct definitions/equivalences, 1 for weak associations.

Use **{record_delimiter}** as the list delimiter.
When finished, output **{completion_delimiter}**

# Examples (Combinatory Logic Specific)

Example 1:
Entities: ["CARTESIAN CLOSED CATEGORIES (CCCS)", "λ-CALCULI", "TYPED λβη-CALCULUS", "TYPE-FREE λ-CALCULUS", "COMBINATORY LOGIC", "LAMBEK", "SCOTT"]
Text:
The connection between Cartesian closed categories (CCCs) and λ-calculi has been thoroughly investigated in the last two decades along two (somewhat) different lines. In the first of these threads, initiated by the pioneering work by Lambek (1980, 1985, 1986), the relations between CCCs and typed λβη-calculus was completely clarified. Scott (1980), Poigné (1986), Curien (1986), Mitchell and Scott (1989) among others, developed the well known isomorphism between CCCs and the typed calculus extended with surjective pairing. The main aim of the second research direction was to characterize the models of the type-free λ-calculus in a categorical setting. Building on early ideas of Scott, Obtulowicz (1987), Berry (1980), Koymans (1982), Longo and Moggi (1990), showed that all the models of the type-free λ-calculus (that is, λ-algebras and λ-models) and Combinatory Logic can be understood as objects enjoying specific properties in a CCC.

Output:
("relationship"{tuple_delimiter}"CARTESIAN CLOSED CATEGORIES (CCCS)"{tuple_delimiter}"λ-CALCULI"{tuple_delimiter}"Are fundamentally connected, with their connection being a major subject of investigation."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"LAMBEK"{tuple_delimiter}"TYPED λβη-CALCULUS"{tuple_delimiter}"Pioneered the work that clarified the relationship between CCCs and the typed λβη-calculus."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"SCOTT"{tuple_delimiter}"TYPE-FREE λ-CALCULUS"{tuple_delimiter}"Contributed early ideas to the characterization of models for the type-free λ-calculus in a categorical setting."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"TYPE-FREE λ-CALCULUS"{tuple_delimiter}"COMBINATORY LOGIC"{tuple_delimiter}"Are related formalisms whose models can both be understood as objects with specific properties within a Cartesian closed category."{tuple_delimiter}8){completion_delimiter}

Example 2:
Entities: ["WEAK CARTESIAN CLOSED CATEGORY (WCCC)", "(β_nat) EQUATION", "(nat-A) EQUATION", "RETRACTION", "A⁻¹"]
Text:
Definition 2.1. A category C is a weak Cartesian closed category (wCCC, for short) iff it is Cartesian and for all a, b, c ∈ Ob_C there exist b^a ∈ Ob_C, eval_{a,b} ∈ C[b^a × a, b] and Λ_c : C[c × a, b] → C[c, b^a] (in Set), which satisfy:
(β_nat) eval_{a,b} ◦ (Λ_c(f) × id_a) = f
(nat-A) Λ_d(g ◦ (h × id_a)) = Λ_c(g) ◦ h
Given a wCCC C, we can define A⁻¹ = eval ◦ <_, id>. The equation (β_nat) then says that A⁻¹(A(f)) = f; however, the lack of (η_nat) prevents us from proving A(A⁻¹(g)) = g, that is, we have just a retraction between the homsets.

Output:
("relationship"{tuple_delimiter}"WEAK CARTESIAN CLOSED CATEGORY (WCCC)"{tuple_delimiter}"(β_nat) EQUATION"{tuple_delimiter}"Is partially defined by the requirement that it must satisfy the (β_nat) equation."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"WEAK CARTESIAN CLOSED CATEGORY (WCCC)"{tuple_delimiter}"(nat-A) EQUATION"{tuple_delimiter}"Is partially defined by the requirement that it must satisfy the (nat-A) equation, which ensures naturality."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"A⁻¹"{tuple_delimiter}"eval"{tuple_delimiter}"Is explicitly defined using the 'eval' morphism as A⁻¹ = eval ◦ <_, id>."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"WEAK CARTESIAN CLOSED CATEGORY (WCCC)"{tuple_delimiter}"RETRACTION"{tuple_delimiter}"The defining properties of a wCCC (specifically, the lack of the (η_nat) equation) result in having a retraction between homsets, rather than a full isomorphism."{tuple_delimiter}8){completion_delimiter}

Example 3:
Entities: ["WCCL (FORMAL THEORY OF WCCC)", "λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)", "TRANSLATION (#.#)", "THEOREM 4.5"]
Text:
To prove the equivalence of wCCL and λβp', we shall define two equality preserving translations, in the spirit of Lambek and Scott (1986) and Curien (1986).
Definition 4.4. (Translation of wCCL into λβp')
For any wCCL-object A, define the λβp'-type #A# as follows: #1# = 1, #A × B# = #A# × #B#, #B^A# = #B#^#A#.
To any wCCL-term f: A → B, define inductively the λβp'-term #f# of type #B#, such that FV(#f#) = {w^A}.
Theorem 4.5. Let f and g be two wCCL-terms. Then wCCL ⊢ f = g ⇒ λβp' ⊢ #f# = #g#

Output:
("relationship"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)"{tuple_delimiter}"Are proven to be equivalent formal systems, connected via equality-preserving translations."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"TRANSLATION (#.#)"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Provides a formal, inductive translation from the terms and types of the wCCL theory."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"TRANSLATION (#.#)"{tuple_delimiter}"λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)"{tuple_delimiter}"Provides a formal, inductive translation into the terms and types of the λβp' calculus."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"THEOREM 4.5"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Formally states that if two wCCL-terms are equal, then their translations into λβp' are also equal, establishing one direction of their equivalence."{tuple_delimiter}9){completion_delimiter}

# Real Data
Entities: {entities}
Text: {input_text}
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are an expert in Combinatory Logic. Your task is to synthesize a single, comprehensive, and rigorous description for a given entity or group of entities.

# Goal
Given an entity name (or a pair of related entities for a relationship) and a list of its descriptions collected from various parts of the knowledge base, combine all information into one coherent, non-contradictory, and detailed description.

# Rules
-   **Comprehensiveness and Completeness**: The final description must be comprehensive and include all relevant information from the provided list. It should, of course, avoid duplications, but it should *not* omit any description, definition, property; even if that means completely repeating the given descriptions. If it seems possible, you can even synthesize new information *if and only if* it is obviously derivable from the existing descriptions.
-   **Contradiction Resolution**: If descriptions contradict, identify the most authoritative or consistent information and resolve the contradiction, explaining the resolution if necessary.
-   **Third Person**: Write the description in the third person.
-   **Full Context**: Include the entity names in the description to provide full context.
-   **Mathematical Precision**: Maintain mathematical precision and rigor.

You should keep in mind that all of the descriptions in the given list will completely be deleted, and your generated description will stand as the only source of information, so you should generate a self-contained, rigorous, comprehensive description that appropriately syntesizes all of the information given.
If the list contains only a few short elements, you can simply merge and replicate them, directly.

# Output Format
Return only the synthesized description string.

# Examples

Example 1:
Entities: "WEAK CARTESIAN CLOSED CATEGORY (WCCC)"
Description List:
- "A category C is a wCCC iff it is Cartesian and for all a, b, c there exist b^a, eval, and Λ which satisfy the equations (β_nat) and (nat-A)."
- "A category C is a wCCC iff for every a, b ∈ Ob_C there exist an object b^a and a natural retraction C[ _ x a, b] < C[ _ , b^a]."
- "These categories correspond to typed Combinatory Logic."
- "In a wCCC, the lack of the (η_nat) equation means we have just a retraction between the homsets, not an isomorphism as in a full CCC."
Output:
A weak Cartesian closed category (wCCC) is a Cartesian category that is formally defined by the existence of an object b^a (exponent) and morphisms 'eval' and 'Λ' for all objects a, b, c, which must satisfy the equations (β_nat) and (nat-A). An equivalent characterization is that for every pair of objects a, b, there exists an exponent object b^a and a natural retraction between the homsets C[ _ x a, b] and C[ _ , b^a]. This structure is weaker than a full Cartesian closed category (CCC) because the lack of the (η_nat) equation results in this retraction rather than a full isomorphism. Semantically, wCCCs provide the categorical structure corresponding to typed Combinatory Logic.

Example 2:
Entities: "RETRACTION"
Description List:
- "In any category C, given a, b ∈ Ob_C, we say that a is a retract of b iff there exist morphisms g ∈ C[a, b] and f ∈ C[b, a] s.t. f ◦ g = id_a."
- "We have just a retraction between the homsets (recall that in a CCC we have an isomorphism)."
- "For F, G: C → D functors, there exists a natural retraction between F and G."
Output:
In a category C, an object 'a' is a retract of an object 'b' if there exist a pair of morphisms, g: a → b and f: b → a, such that their composition in one direction, f ◦ g, is equal to the identity morphism on 'a' (id_a). This concept is weaker than an isomorphism because the reverse composition, g ◦ f, is not required to be an identity. The notion of retraction is general and can also be applied between functors, where it is called a natural retraction.

Example 3:
Entities: "PL-CATEGORY"
Description List:
- "A PL-category (PL stands for Polymorphic Language) (S, G) is given by: (i) a CCC S (global category), with a distinguished object Ω ∈ Ob_S."
- "A PL-category is also given by: (ii) an explicit indexed category over S, that is a functor G: S^op → Cat, such that for any object A of S, G(A) (local category) is a CCC."
Output:
A PL-category, which stands for Polymorphic Language category, is a structure denoted by the pair (S, G) that is composed of two primary components. The first component is a Cartesian closed category (CCC) named S, which is referred to as the 'global category' and must contain a distinguished object Ω. The second component is an explicit indexed category over S, which is represented by a functor G from the opposite category of S to the category of categories (Cat), such that for any object A in the global category S, the resulting G(A) is itself a CCC, referred to as a 'local category'.

# Real Data
Entities: {entity_name}
Description List: {description_list}
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """It appears some entities were missed in the last extraction. Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Based on the remaining text, are there any more factual entities (postulates, objects, concepts, properties, or proofs) that need to be extracted according to the strict rules provided previously? Answer YES | NO.
"""

PROMPTS[
    "summary_clusters"
] = """You are an expert in Combinatory Logic, functioning as a knowledge architect. Your task is to organize a given cluster of related entities into a more structured hierarchy. You will achieve this by synthesizing one or more *new, higher-level* entities that abstract or encompass the concepts presented in the input cluster.

# Goal
Given a list of entity descriptions from a conceptual cluster, identify and define one or more new, higher-level attribute entities that represent a unifying theme, object, or postulate. These new entities will form the next layer in a hierarchical knowledge graph.

# Rules
1.  **Synthesize, Don't Just Group**: Your primary task is to identify a unifying theme or a more general concept that the entities in the cluster are instances of, or components of. The new entity should provide a higher level of understanding.
2.  **Abstraction, Not Redundancy**: The new entity must represent a genuine abstraction. It should not be a mere synonym or restatement of an existing non-temporary entity in the cluster. For example, if the cluster contains "I combinator" and "K combinator", a good summary entity is "Basic Combinators", not "Identity and Constant Combinators".
3.  **Handling of Temporary Entities**: While the input cluster may contain temporary entities, your synthesized higher-level entity **must always be non-temporary**. You should use the temporary entities as clues to understand the context, but the final summary entity must represent a stable, reusable concept.
4.  **Multiple Summaries for Complex Clusters**: If the input cluster is complex and appears to represent multiple distinct high-level ideas or systems, you are encouraged to generate more than one attribute entity to capture this nuance.
5.  **Relationship Creation**: After defining the new higher-level entity (or entities), you must create relationships linking the relevant *original non-temporary entities* from the input list to the new entity you just created. This is just as important and the relationships should be robust, well-written and high-quality.

# Entity Types for Summarization
Your summarized attribute entity must fit one of these types: [{meta_attribute_list}].

# Output Format
Return output in English as a single list of the newly identified attribute entities and their relationships to the original entities.
Format each entity as:
`("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<is_temporary>)`
Format each relationship as:
`("relationship"{tuple_delimiter}<source_entity_name>{tuple_delimiter}<target_entity_name>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)`

Use **{record_delimiter}** as the list delimiter.
When finished, output **{completion_delimiter}**

# Examples (Combinatory Logic Specific)

Example 1:
Meta attribute list: ["theory", "system", "postulate", "proof", "concept", "object", "property"]
Entity description list:
- ("id_A", "The identity morphism on object A.", false)
- ("fst_A,B", "The first projection morphism from a product A x B to A.", false)
- ("snd_A,B", "The second projection morphism from a product A x B to B.", false)
- ("eval_A,B", "The evaluation morphism for an exponent object.", false)
- ("<f,g>", "The pairing constructor for morphisms.", false)
- ("A(f)", "The currying or abstraction operator for morphisms.", false)
Output:
("entity"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"theory"{tuple_delimiter}"A formal theory for weak Cartesian closed categories, defined equationally by a set of objects (1, A x B, B^A) and a signature of fundamental term-forming morphisms, including identity, projections, evaluation, pairing, and abstraction."{tuple_delimiter}false){record_delimiter}
("relationship"{tuple_delimiter}"id_A"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Is the identity term constructor in the wCCL theory."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"fst_A,B"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Is the first projection term constructor in the wCCL theory."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"eval_A,B"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Is the evaluation term constructor in the wCCL theory."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"A(f)"{tuple_delimiter}"WCCL (FORMAL THEORY OF WCCC)"{tuple_delimiter}"Is the abstraction (currying) term constructor in the wCCL theory."{tuple_delimiter}10){completion_delimiter}

Example 2:
Meta attribute list: ["theory", "system", "postulate", "proof", "concept", "object", "property"]
Entity description list:
- ("λ-ALGEBRA", "A model of the type-free λ-calculus.", false)
- ("λ-MODEL", "A model of the type-form λ-calculus.", false)
- ("COMBINATORY LOGIC MODEL", "A model for Combinatory Logic, given by a principal morphism and retractions.", false)
Output:
("entity"{tuple_delimiter}"MODELS OF TYPE-FREE CALCULI"{tuple_delimiter}"theory"{tuple_delimiter}"A general class of models for type-free calculi, such as λ-algebras, λ-models, and models of Combinatory Logic, which can be understood as objects with specific properties within a Cartesian closed category."{tuple_delimiter}false){record_delimiter}
("relationship"{tuple_delimiter}"λ-ALGEBRA"{tuple_delimiter}"MODELS OF TYPE-FREE CALCULI"{tuple_delimiter}"Is a specific kind of model for type-free calculi."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"λ-MODEL"{tuple_delimiter}"MODELS OF TYPE-FREE CALCULI"{tuple_delimiter}"Is a specific kind of model for type-free calculi."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"COMBINATORY LOGIC MODEL"{tuple_delimiter}"MODELS OF TYPE-FREE CALCULI"{tuple_delimiter}"Is a specific kind of model for type-free calculi."{tuple_delimiter}9){completion_delimiter}

Example 3:
Meta attribute list: ["theory", "system", "postulate", "proof", "concept", "object", "property"]
Entity description list:
- ("K_σ,τ", "The K combinator for specific types.", false)
- ("S_σ,τ,ρ", "The S combinator for specific types.", false)
- ("fst_σ,τ", "The first projection constructor.", false)
- ("<M,N>", "The pairing constructor for terms.", false)
- ("λx:σ.M", "The lambda abstraction constructor.", false)
- ("MN", "The application of term M to term N.", false)
Output:
("entity"{tuple_delimiter}"TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING (λβp')"{tuple_delimiter}"system"{tuple_delimiter}"A formal system defined by types, terms, and conversion rules. Its term constructors include variables, application (MN), lambda abstraction (λx:σ.M), pairing (<M,N>), and projections (fst, snd)."{tuple_delimiter}false){record_delimiter}
("entity"{tuple_delimiter}"TYPED COMBINATORY LOGIC (CLp')"{tuple_delimiter}"system"{tuple_delimiter}"A formal system defined by types, terms, and conversion rules. Its term constructors include variables, application (MN), and the basic combinators K, S, fst, snd, and D."{tuple_delimiter}false){record_delimiter}
("relationship"{tuple_delimiter}"<M,N>"{tuple_delimiter}"TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING (λβp')"{tuple_delimiter}"Is the term constructor for pairing in the λβp' system."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"λx:σ.M"{tuple_delimiter}"TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING (λβp')"{tuple_delimiter}"Is the term constructor for abstraction in the λβp' system."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"K_σ,τ"{tuple_delimiter}"TYPED COMBINATORY LOGIC (CLp')"{tuple_delimiter}"Is a basic combinator and term constructor in the CLp' system."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"S_σ,τ,ρ"{tuple_delimiter}"TYPED COMBINATORY LOGIC (CLp')"{tuple_delimiter}"Is a basic combinator and term constructor in the CLp' system."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"MN"{tuple_delimiter}"TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING (λβp')"{tuple_delimiter}"Is the term constructor for application, common to both systems."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"MN"{tuple_delimiter}"TYPED COMBINATORY LOGIC (CLp')"{tuple_delimiter}"Is the term constructor for application, common to both systems."{tuple_delimiter}8){completion_delimiter}

# Real Data
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}
Output:
"""

# --- New Prompt for Relationship Refinement ---

# PROMPTS[
#     "refine_relationship_context"
# ] = """You are an exceptionally precise and rigorous expert in Combinatory Logic (CL). Your task is to refine the description of a relationship between two entities.

# # Goal
# Given two related entities (source and target), a simple description of their relationship, and the original text from which this relationship was extracted, rewrite the `relationship_description` to be more general, self-contained, and mathematically precise. The goal is to capture the underlying logical reason, property, or principle that establishes this relationship.

# # Rules for Refinement
# 1.  **Capture the "Why"**: If the original text implies *why* the relationship holds (e.g., "due to property X", "by definition", "as a consequence of theorem Y"), explicitly incorporate this reason into the refined description.
# 2.  **Generalize**: Replace local, temporary names (e.g., "M", "the combinator defined above") with their defining characteristics or the general class they belong to, *if* doing so makes the relationship more universally understandable without losing precision. Refer to the `entity_description` of the source and target entities for this purpose.
# 3.  **Self-Contained**: The refined `relationship_description` should be understandable on its own, without needing to refer back to the original text or the specific names of temporary entities.
# 4.  **Mathematical Precision**: Use precise mathematical language.
# 5.  **No New Entities**: Do not extract or define new entities in the refined description. Focus solely on describing the relationship between the *given* source and target entities.

# # Output Format
# Return only the refined `relationship_description` string.

# # Examples

# Example 1:
# - Source Entity: {"name": "M", "description": "A temporary combinator defined as SKK."}
# - Target Entity: {"name": "N", "description": "A temporary combinator that is equivalent to M."}
# - Simple Relationship: "M is equivalent to N"
# - Original Text: "...Thus, we see that M is equivalent to N, based on the principle of weak reduction."
# Output:
# A combinator defined as SKK is equivalent to a combinator that is equivalent to SKK, a relationship established by the principle of weak reduction.

# Example 2:
# - Source Entity: {"name": "CL_I", "description": "The combinatory logic system based on the I combinator."}
# - Target Entity: {"name": "CL_K", "description": "The combinatory logic system based on the K combinator."}
# - Simple Relationship: "CL_I is a subset of CL_K"
# - Original Text: "It can be shown that CL_I is a subset of CL_K, as any term expressible in CL_I can also be expressed in CL_K by using the K combinator to simulate the identity."
# Output:
# The combinatory logic system based on the I combinator is a subset of the combinatory logic system based on the K combinator, because any term expressible in CL_I can also be expressed in CL_K by simulating the identity using the K combinator.

# Example 3:
# - Source Entity: {"name": "CHURCH-ROSSER THEOREM", "description": "A fundamental theorem ensuring the uniqueness of normal forms."}
# - Target Entity: {"name": "CONFLUENCE", "description": "A property indicating that reduction order does not affect the final normal form."}
# - Simple Relationship: "Church-Rosser Theorem implies Confluence"
# - Original Text: "The Church-Rosser Theorem directly implies the property of confluence, as it guarantees that all reduction paths from a given term lead to the same normal form."
# Output:
# The Church-Rosser Theorem directly implies the property of confluence, as it guarantees that all reduction paths from a given term lead to the same normal form, thereby ensuring the uniqueness of normal forms.

# # Real Data
# - Source Entity: {source_entity}
# - Target Entity: {target_entity}
# - Simple Relationship: {relation_description}
# - Original Text: {context_text}
# Output:
# """

# --- General Purpose Prompts (Adjusted for CL context) ---

PROMPTS[
    "community_report"
] = """You are an AI assistant with deep expertise in mathematical logic, specializing in Combinatory Logic and Category Theory. Your task is to function as a research analyst. You will be given a "community" – a cluster of interconnected entities (postulates, objects, concepts, etc.) and their relationships, all extracted from a formal text. Your goal is to write a comprehensive, insightful, and structured report on this community.

# Goal
Write a formal report on the provided community of entities. This report should elucidate the core theme of the community, explain the roles of its key components, and detail the significant logical connections and implications. The report is intended for a researcher who wants to quickly grasp the essence and importance of this specific conceptual area.

# Report Structure
Your output must be a single, well-formed JSON object containing the following sections:

-   **title**: A concise, descriptive, and academic title for the community. The title must represent the central, non-temporary concepts being discussed.
-   **summary**: A dense, executive summary (2-4 sentences). This should explain the community's core purpose, the interplay between its main non-temporary entities, and its overall significance in the broader field.
-   **importance_rating**: A float score from 0.0 to 10.0 representing the conceptual importance and centrality of this community to the field of Combinatory Logic or Category Theory. A high score indicates a foundational or highly impactful set of ideas.
-   **rating_explanation**: A single, precise sentence justifying the `importance_rating`.
-   **detailed_findings**: An array of 3-5 key insights about the community. Each finding must be an object with two keys:
    -   `summary`: A short, one-sentence summary of the insight.
    -   `explanation`: A detailed, multi-sentence paragraph that elaborates on the insight. This explanation must be grounded in the provided data, be mathematically precise, and explain the "why" and "how" of the relationships between entities. Prioritize insights derived from non-temporary entities.

# Grounding and Quality Rules
1.  **Evidence-Based**: Every statement in your report must be directly supported by the information in the provided `Entities` and `Relationships` tables. Do not introduce outside knowledge or speculate.
2.  **Focus on Non-Temporary Entities**: While temporary entities might be present in the data, your report's title, summary, and findings should focus on the stable, non-temporary concepts. Mention temporary entities only if they are essential to explaining a key relationship involving a non-temporary one.
3.  **Mathematical Rigor**: Use precise, formal language appropriate for a mathematical research context.
4.  **Logical Flow**: The `detailed_findings` should tell a story, building upon each other to explain the community's structure, from its foundational definitions to its main results and implications.

# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description,is_temporary
1,WEAK CARTESIAN CLOSED CATEGORY (WCCC),object,"A category with finite products and a weaker form of exponents, defined by (β_nat) and (nat-A) equations.",false
2,(β_nat) EQUATION,object,"The equation eval ◦ (Λ(f) × id) = f, which is a defining property of a wCCC.",false
3,(nat-A) EQUATION,object,"The equation Λ(g ◦ (h × id)) = Λ(g) ◦ h, which ensures the naturality of the Λ operator.",false
4,RETRACTION,concept,"A relationship between two objects where f ◦ g = id, but not necessarily g ◦ f = id. It is weaker than an isomorphism.",false
5,TYPED COMBINATORY LOGIC,system,"A formal system of logic based on combinators with types.",false
6,A⁻¹,object,"A temporary object defined as eval ◦ <_, id> within the context of a wCCC.",true
```
Relationships:
```csv
id,source,target,description,weight
101,WEAK CARTESIAN CLOSED CATEGORY (WCCC),(β_nat) EQUATION,"Is partially defined by the requirement that it must satisfy the (β_nat) equation.",10
102,WEAK CARTESIAN CLOSED CATEGORY (WCCC),(nat-A) EQUATION,"Is partially defined by the requirement that it must satisfy the (nat-A) equation.",10
103,WEAK CARTESIAN CLOSED CATEGORY (WCCC),RETRACTION,"The defining properties of a wCCC result in having a retraction between homsets, not a full isomorphism.",8
104,WEAK CARTESIAN CLOSED CATEGORY (WCCC),TYPED COMBINATORY LOGIC,"wCCCs provide the categorical semantics that correspond to typed Combinatory Logic.",9
105,A⁻¹,(β_nat) EQUATION,"The (β_nat) equation implies that A⁻¹(A(f)) = f, demonstrating the retraction property.",8
``````
Output:
```json
{
    "title": "Weak Cartesian Closed Categories (wCCCs) as a Model for Typed Combinatory Logic",
    "summary": "This community defines the structure of a weak Cartesian closed category (wCCC) through its core definitional equations, (β_nat) and (nat-A). It establishes that this weaker categorical structure, which results in a retraction rather than a full isomorphism for homsets, serves as the precise categorical semantics for typed Combinatory Logic.",
    "importance_rating": 9.0,
    "rating_explanation": "This community is highly important as it formally connects the syntactic world of typed logical systems with the semantic world of category theory.",
    "detailed_findings": [
        {
            "summary": "A wCCC is defined by retaining naturality while dropping the uniqueness of exponents.",
            "explanation": "A weak Cartesian closed category (wCCC) is a modification of a standard CCC. It is formally constructed by requiring the satisfaction of two key equations: (β_nat), which ensures evaluation behaves correctly, and (nat-A), which preserves the crucial property of naturality for the abstraction operator Λ. This structure intentionally omits the (η_nat) equation, which would enforce uniqueness and lead to a full isomorphism."
        },
        {
            "summary": "The structure of a wCCC implies a retraction, not an isomorphism, between homsets.",
            "explanation": "A direct consequence of the wCCC's defining axioms is that the relationship between the homset C[c x a, b] and the exponent object's homset C[c, b^a] is merely a retraction. This means that while one can map from a function to its curried form and back again (A⁻¹(A(f)) = f), the reverse is not guaranteed. This is a fundamental distinction from a full CCC, where this relationship is an isomorphism."
        },
        {
            "summary": "wCCCs provide the exact categorical semantics for typed Combinatory Logic.",
            "explanation": "The primary significance of the wCCC structure is its direct correspondence to typed Combinatory Logic. The specific properties of a wCCC—having products and a form of exponents that support evaluation and natural abstraction but are not fully unique—perfectly model the behavior of a typed CL system. This provides a robust, abstract semantic foundation for the logic."
        }
    ]
}
```

# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

Your output must be a single, well-formed JSON object.

Output:
"""

PROMPTS["local_rag_response"] = ""  # TODO

PROMPTS["naive_rag_response"] = ""  # TODO

PROMPTS["fail_response"] = (
    "Sorry, I am not able to provide an answer to that question based on the available knowledge."
)

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

# --- Combinatory Logic Specific Entity Types ---
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "postulate",  # theorem, lemma, proposition, corollary, remark, asserted claim
    "object",  # term, system, theory, model, axiom, definition, function, map, category, logic, relation, set, element, rule
    "concept",  # abstraction, reduction, normal form, consistency, completeness
    "property",  # confluence, extensionality, strong normalization
    "proof",  # a formal argument or derivation
]

# --- Meta Entity Types for Hierarchical Summarization ---
PROMPTS["META_ENTITY_TYPES"] = [
    "theory",  # a formal system or framework
    "system",  # a structured set of concepts or components
    "postulate",  # a fundamental assumption or principle
    "proof",  # a formal argument establishing the truth of a statement
    "concept",  # an abstract idea or general notion
    "object",  # a specific entity or instance within the theory
    "property",  # a characteristic or attribute of an object or concept
]

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "<|RECORD|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# --- Entity Disambiguation Prompt ---
PROMPTS["entity_disambiguation"] = """
You are a meticulous knowledge graph curator and an expert in Combinatory Logic. Your task is to perform entity disambiguation. You will be given a cluster of entities that are suspected to be aliases for the same underlying concept. Your job is to analyze the provided evidence and make a definitive judgment.

# Goal
Determine if the entities in the provided list are aliases for one another. If they are, you must select the best canonical name. If they are not, you must state that they should not be merged.

# Rules of Judgment
1.  **Identity vs. Similarity**: Do not merge entities that are merely similar. Merge only if you are highly confident they refer to the exact same concept, postulate, or object. Subtle differences in definitions matter.
2.  **Context is Ground Truth**: The `original_text_context` is the most important piece of evidence. An entity's meaning is defined by its use in the source document. Your justification MUST reference specific phrases from this context.
3.  **Be Conservative**: If there is any ambiguity or insufficient evidence to prove the entities are identical, you MUST decide "DO_NOT_MERGE". It is better to leave two aliases separate than to incorrectly merge two distinct concepts.
4.  **Canonical Name Selection**: If you decide to merge, the canonical name should be the most precise and commonly used term. Prefer formal, shorter names (e.g., "CL_η") over descriptive, longer ones (e.g., "The theory CL_η").

# Input Data
You will be provided with a JSON list of candidate entities. Each entity object has the following structure:
{{
  "entity_name": "The name extracted for this entity.",
  "entity_description": "The description generated for this entity.",
  "entity_type": "The type of entity (postulate, object, concept, property, proof).",
  "original_text_context": "The full text chunk from which this entity was extracted."
}}

# Output Format
You MUST respond with a single, well-formed JSON object with the following structure. Do not add any text outside of this JSON object.

- For a MERGE decision:
{{
  "decision": "MERGE",
  "canonical_name": "<The chosen canonical name>",
  "aliases": ["<list of other names to be merged>"],
  "confidence_score": <A float from 0.0 to 1.0 indicating your confidence in this decision>,
  "justification": "A detailed explanation for why these entities are identical, referencing specific evidence from the provided context."
}}

- For a DO_NOT_MERGE decision:
{{
  "decision": "DO_NOT_MERGE",
  "confidence_score": <A float from 0.0 to 1.0 indicating your confidence in this decision>,
  "justification": "A detailed explanation of the subtle differences that prevent these entities from being merged, referencing specific evidence from the provided context."
}}

# Real Data

Candidate Cluster:
{input_json_for_cluster}

# Your Decision:
"""
