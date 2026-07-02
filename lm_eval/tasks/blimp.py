"""
BLiMP: A Benchmark of Linguistic Minimal Pairs for English
https://arxiv.org/abs/1912.00582

BLiMP is a challenge set for evaluating what language models (LMs) know about
major grammatical phenomena in English. BLiMP consists of 67 sub-datasets, each
containing 1000 minimal pairs isolating specific contrasts in syntax, morphology,
or semantics. The data is automatically generated according to expert-crafted
grammars.

Homepage: https://github.com/alexwarstadt/blimp
"""
from lm_eval.api.task import PromptSourceTask
from typing import Optional, List
from datasets import load_dataset

import string


_CITATION = """
@article{warstadt2019blimp,
    author = {Warstadt, Alex and Parrish, Alicia and Liu, Haokun and Mohananey, Anhad and Peng, Wei and Wang, Sheng-Fu and Bowman, Samuel R.},
    title = {BLiMP: The Benchmark of Linguistic Minimal Pairs for English},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {8},
    number = {},
    pages = {377-392},
    year = {2020},
    doi = {10.1162/tacla00321},
    URL = {https://doi.org/10.1162/tacl_a_00321},
    eprint = {https://doi.org/10.1162/tacl_a_00321},
    abstract = { We introduce The Benchmark of Linguistic Minimal Pairs (BLiMP),1 a challenge set for evaluating the linguistic knowledge of language models (LMs) on major grammatical phenomena in English. BLiMP consists of 67 individual datasets, each containing 1,000 minimal pairs—that is, pairs of minimally different sentences that contrast in grammatical acceptability and isolate specific phenomenon in syntax, morphology, or semantics. We generate the data according to linguist-crafted grammar templates, and human aggregate agreement with the labels is 96.4. We evaluate n-gram, LSTM, and Transformer (GPT-2 and Transformer-XL) LMs by observing whether they assign a higher probability to the acceptable sentence in each minimal pair. We find that state-of-the-art models identify morphological contrasts related to agreement reliably, but they struggle with some subtle semantic and syntactic phenomena, such as negative polarity items and extraction islands. }
}
"""


class BlimpTask(PromptSourceTask):
    DATASET_PATH = "blimp"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        # The HF dataset only contains a "train" dataset, but the harness expects a "validation"
        # dataset. Let's use the training dataset, on the assumption that the model wasn't actually
        # trained on this data.
        return self.dataset["train"]

    def null_prompt_doc_to_text(self, doc: dict) -> str:
        return ""

    def null_prompt_doc_to_target(self, doc: dict) -> List[str]:
        correct_choice = doc["sentence_good"]
        return [correct_choice]

    def null_prompt_answer_choices(self, doc: dict) -> List[str]:
        choices = [doc["sentence_good"], doc["sentence_bad"]]
        return choices


    # def _calculate_token_distance(self, sentence: str, controller: str, target: str, tokenizer):
    #     """Calculate token distance between controller and target using the tokenizer."""
    #     try:
    #         # Encode the full sentence
    #         sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    #         sentence_token_strings = tokenizer.convert_ids_to_tokens(sentence_tokens)
            
    #         controller_lower = controller.lower().strip()
    #         target_lower = target.lower().strip()
            
    #         # Reconstruct words from token strings to handle subword tokens
    #         current_word = ""
    #         current_word_length = 0
    #         current_word_start = 0  # Track where current word starts
    #         word_list = []
    #         word_start_indices = []
    #         word_lengths = []
            
    #         for i, token_str in enumerate(sentence_token_strings):
    #             if token_str.startswith("Ġ"):
    #                 # Save previous word if it exists
    #                 if current_word:
    #                     word_list.append(current_word)
    #                     word_start_indices.append(current_word_start)
    #                     word_lengths.append(current_word_length)
    #                 # Start new word
    #                 current_word = token_str.replace("Ġ", "").lower()
    #                 current_word_length = 1
    #                 current_word_start = i  # Record start of this word
    #             else:
    #                 current_word += token_str.lower()
    #                 current_word_length += 1
            
    #         # Don't forget the last word
    #         if current_word:
    #             word_list.append(current_word)
    #             word_start_indices.append(current_word_start)
    #             word_lengths.append(current_word_length)
            
    #         # Strip punctuation from words for matching
    #         word_list_clean = [w.strip(string.punctuation) for w in word_list]
            
    #         # Find controller and target in word list
    #         controller_words = controller_lower.split()
    #         target_words = target_lower.split()
            
    #         # ===== Extract only the last word of multi-word controller =====
    #         controller_search = controller_words[-1] if len(controller_words) > 0 else ""
            
    #         controller_end_idx = -1
    #         target_idx = -1
            
    #         for i in range(len(word_list_clean)):
    #             if word_list_clean[i] == controller_search:
    #                 controller_end_idx = word_start_indices[i] + word_lengths[i]
    #                 break
    #         # ==========
            
    #         # Find multi-word target
    #         for i in range(len(word_list_clean) - len(target_words) + 1):
    #             if word_list_clean[i:i+len(target_words)] == target_words:
    #                 target_idx = word_start_indices[i]
    #                 break
            
    #         if controller_end_idx >= 0 and target_idx >= 0:
    #             token_distance = abs(target_idx - controller_end_idx) + 1
    #             return token_distance
    #         else:
    #             return -1
    #     except Exception as e:
    #         print(f"Error: {e}\n")
    #         import traceback
    #         traceback.print_exc()
    #         return -1
    #     ####################


    import string
    def _calculate_token_distance(self, sentence: str, controller: str, target: str, tokenizer):
        """Calculate token distance between controller and target using the tokenizer."""
        try:
            # Encode the full sentence
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_token_strings = tokenizer.convert_ids_to_tokens(sentence_tokens)
            
            controller_lower = controller.lower().strip()
            target_lower = target.lower().strip()
            
            # Reconstruct words from token strings to handle subword tokens
            current_word = ""
            current_word_length = 0
            current_word_start = 0  # Track where current word starts
            word_list = []
            word_start_indices = []
            word_lengths = []
            
            for i, token_str in enumerate(sentence_token_strings):
                if token_str.startswith("Ġ"):
                    # Save previous word if it exists
                    if current_word:
                        word_list.append(current_word)
                        word_start_indices.append(current_word_start)
                        word_lengths.append(current_word_length)
                    # Start new word
                    current_word = token_str.replace("Ġ", "").lower()
                    current_word_length = 1
                    current_word_start = i  # Record start of this word
                else:
                    current_word += token_str.lower()
                    current_word_length += 1
            
            # Don't forget the last word
            if current_word:
                word_list.append(current_word)
                word_start_indices.append(current_word_start)
                word_lengths.append(current_word_length)
            
            # Strip punctuation from words for matching
            word_list_clean = [w.strip(string.punctuation) for w in word_list]
            
            # Find controller and target in word list
            controller_words = controller_lower.split()
            target_words = target_lower.split()
            
            # ===== Extract only the last word of multi-word controller =====
            controller_search = controller_words[-1] if len(controller_words) > 0 else ""
            
            controller_end_idx = -1
            target_idx = -1
            
            for i in range(len(word_list_clean)):
                if word_list_clean[i] == controller_search:
                    controller_end_idx = word_start_indices[i] + word_lengths[i]
            # ==========
            
            # Find the LAST occurrence of multi-word target
            for i in range(len(word_list_clean) - len(target_words), -1, -1):
                if word_list_clean[i:i+len(target_words)] == target_words:
                    target_idx = word_start_indices[i]
                    break
            
            if controller_end_idx >= 0 and target_idx >= 0:
                token_distance = abs(target_idx - controller_end_idx) + 1
                return token_distance
            else:
                return -1
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
            return -1

    def null_prompt_get_logging_info(self):
        return {
            "fixed_answer_choice_list": None,
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "subset": self.SPLIT,
            "prompt_name": None,
            "prompt_id": None,
            "prompt_jinja": None,
            "prompt_original_task": f"{self.DATASET_PATH}/{self.DATASET_NAME}",
            # Placeholder for comment in post-processing.
            "comment": "",
        }

class BlimpAdjunctIsland(BlimpTask):
    DATASET_NAME = "adjunct_island"


class BlimpAnaphorGenderAgreement(BlimpTask):
    DATASET_NAME = "anaphor_gender_agreement"


class BlimpAnaphorNumberAgreement(BlimpTask):
    DATASET_NAME = "anaphor_number_agreement"


class BlimpAnimateSubjectPassive(BlimpTask):
    DATASET_NAME = "animate_subject_passive"


class BlimpAnimateSubjectTrans(BlimpTask):
    DATASET_NAME = "animate_subject_trans"


class BlimpCausative(BlimpTask):
    DATASET_NAME = "causative"


class BlimpComplex_NPIsland(BlimpTask):
    DATASET_NAME = "complex_NP_island"


class BlimpCoordinateStructureConstraintComplexLeftBranch(BlimpTask):
    DATASET_NAME = "coordinate_structure_constraint_complex_left_branch"


class BlimpCoordinateStructureConstraintObjectExtraction(BlimpTask):
    DATASET_NAME = "coordinate_structure_constraint_object_extraction"


class BlimpDeterminerNounAgreement_1(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_1"


class BlimpDeterminerNounAgreement_2(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_2"


class BlimpDeterminerNounAgreementIrregular_1(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_irregular_1"


class BlimpDeterminerNounAgreementIrregular_2(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_irregular_2"


class BlimpDeterminerNounAgreementWithAdj_2(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_with_adj_2"


class BlimpDeterminerNounAgreementWithAdjIrregular_1(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_with_adj_irregular_1"


class BlimpDeterminerNounAgreementWithAdjIrregular_2(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_with_adj_irregular_2"


class BlimpDeterminerNounAgreementWithAdjective_1(BlimpTask):
    DATASET_NAME = "determiner_noun_agreement_with_adjective_1"


class BlimpDistractorAgreementRelationalNoun(BlimpTask):
    DATASET_NAME = "distractor_agreement_relational_noun"


class BlimpDistractorAgreementRelativeClause(BlimpTask):
    DATASET_NAME = "distractor_agreement_relative_clause"


class BlimpDropArgument(BlimpTask):
    DATASET_NAME = "drop_argument"


class BlimpEllipsisNBar_1(BlimpTask):
    DATASET_NAME = "ellipsis_n_bar_1"


class BlimpEllipsisNBar_2(BlimpTask):
    DATASET_NAME = "ellipsis_n_bar_2"


class BlimpExistentialThereObjectRaising(BlimpTask):
    DATASET_NAME = "existential_there_object_raising"


class BlimpExistentialThereQuantifiers_1(BlimpTask):
    DATASET_NAME = "existential_there_quantifiers_1"


class BlimpExistentialThereQuantifiers_2(BlimpTask):
    DATASET_NAME = "existential_there_quantifiers_2"


class BlimpExistentialThereSubjectRaising(BlimpTask):
    DATASET_NAME = "existential_there_subject_raising"


class BlimpExpletiveItObjectRaising(BlimpTask):
    DATASET_NAME = "expletive_it_object_raising"


class BlimpInchoative(BlimpTask):
    DATASET_NAME = "inchoative"


class BlimpIntransitive(BlimpTask):
    DATASET_NAME = "intransitive"


class BlimpIrregularPastParticipleAdjectives(BlimpTask):
    DATASET_NAME = "irregular_past_participle_adjectives"


class BlimpIrregularPastParticipleVerbs(BlimpTask):
    DATASET_NAME = "irregular_past_participle_verbs"


class BlimpIrregularPluralSubjectVerbAgreement_1(BlimpTask):
    DATASET_NAME = "irregular_plural_subject_verb_agreement_1"


class BlimpIrregularPluralSubjectVerbAgreement_2(BlimpTask):
    DATASET_NAME = "irregular_plural_subject_verb_agreement_2"


class BlimpLeftBranchIslandEchoQuestion(BlimpTask):
    DATASET_NAME = "left_branch_island_echo_question"


class BlimpLeftBranchIslandSimpleQuestion(BlimpTask):
    DATASET_NAME = "left_branch_island_simple_question"


class BlimpMatrixQuestionNpiLicensorPresent(BlimpTask):
    DATASET_NAME = "matrix_question_npi_licensor_present"


class BlimpNpiPresent_1(BlimpTask):
    DATASET_NAME = "npi_present_1"


class BlimpNpiPresent_2(BlimpTask):
    DATASET_NAME = "npi_present_2"


class BlimpOnlyNpiLicensorPresent(BlimpTask):
    DATASET_NAME = "only_npi_licensor_present"


class BlimpOnlyNpiScope(BlimpTask):
    DATASET_NAME = "only_npi_scope"


class BlimpPassive_1(BlimpTask):
    DATASET_NAME = "passive_1"


class BlimpPassive_2(BlimpTask):
    DATASET_NAME = "passive_2"


class BlimpPrinciple_ACCommand(BlimpTask):
    DATASET_NAME = "principle_A_c_command"


class BlimpPrinciple_ACase_1(BlimpTask):
    DATASET_NAME = "principle_A_case_1"


class BlimpPrinciple_ACase_2(BlimpTask):
    DATASET_NAME = "principle_A_case_2"


class BlimpPrinciple_ADomain_1(BlimpTask):
    DATASET_NAME = "principle_A_domain_1"


class BlimpPrinciple_ADomain_2(BlimpTask):
    DATASET_NAME = "principle_A_domain_2"


class BlimpPrinciple_ADomain_3(BlimpTask):
    DATASET_NAME = "principle_A_domain_3"


class BlimpPrinciple_AReconstruction(BlimpTask):
    DATASET_NAME = "principle_A_reconstruction"


class BlimpRegularPluralSubjectVerbAgreement_1(BlimpTask):
    DATASET_NAME = "regular_plural_subject_verb_agreement_1"


class BlimpRegularPluralSubjectVerbAgreement_2(BlimpTask):
    DATASET_NAME = "regular_plural_subject_verb_agreement_2"


class BlimpSententialNegationNpiLicensorPresent(BlimpTask):
    DATASET_NAME = "sentential_negation_npi_licensor_present"


class BlimpSententialNegationNpiScope(BlimpTask):
    DATASET_NAME = "sentential_negation_npi_scope"


class BlimpSententialSubjectIsland(BlimpTask):
    DATASET_NAME = "sentential_subject_island"


class BlimpSuperlativeQuantifiers_1(BlimpTask):
    DATASET_NAME = "superlative_quantifiers_1"


class BlimpSuperlativeQuantifiers_2(BlimpTask):
    DATASET_NAME = "superlative_quantifiers_2"


class BlimpToughVsRaising_1(BlimpTask):
    DATASET_NAME = "tough_vs_raising_1"


class BlimpToughVsRaising_2(BlimpTask):
    DATASET_NAME = "tough_vs_raising_2"


class BlimpTransitive(BlimpTask):
    DATASET_NAME = "transitive"


class BlimpWhIsland(BlimpTask):
    DATASET_NAME = "wh_island"


class BlimpWhQuestionsObjectGap(BlimpTask):
    DATASET_NAME = "wh_questions_object_gap"


class BlimpWhQuestionsSubjectGap(BlimpTask):
    DATASET_NAME = "wh_questions_subject_gap"


class BlimpWhQuestionsSubjectGapLongDistance(BlimpTask):
    DATASET_NAME = "wh_questions_subject_gap_long_distance"


class BlimpWhVsThatNoGap(BlimpTask):
    DATASET_NAME = "wh_vs_that_no_gap"


class BlimpWhVsThatNoGapLongDistance(BlimpTask):
    DATASET_NAME = "wh_vs_that_no_gap_long_distance"


class BlimpWhVsThatWithGap(BlimpTask):
    DATASET_NAME = "wh_vs_that_with_gap"


class BlimpWhVsThatWithGapLongDistance(BlimpTask):
    DATASET_NAME = "wh_vs_that_with_gap_long_distance"

class BlimpFromFile(BlimpTask):
    DATASET_NAME = "causative"
