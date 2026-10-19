import argparse
import lm_eval
import os
import json

TASKS = {
    "blimp": ["anaphor_agreement.json", "argument_structure.json", "binding.json",
              "control_raising.json", "determiner_noun_agreement.json", "ellipsis.json",
              "filler_gap.json", "irregular_forms.json", "island_effects.json",
              "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json"],
    "blimp_test": ["control_raising.json","binding.json","anaphor_agreement.json", "argument_structure.json"],
    "supplement": ["hypernym.json", "qa_congruence_easy.json", "qa_congruence_tricky.json",
               "subject_aux_inversion.json", "turn_taking.json"],
    "dep_dist": [
       'anaphor_agreement_adj_1.jsonl', 'anaphor_agreement_adj_2.jsonl',
       'anaphor_agreement_adj_3.jsonl', 'anaphor_agreement_adj_4.jsonl',
       'anaphor_agreement_adj_5.jsonl', 'anaphor_agreement_adj_6.jsonl',
       'anaphor_agreement_adj_7.jsonl', 
       'anaphor_agreement_adv_1.jsonl',
       'anaphor_agreement_adv_2.jsonl', 'anaphor_agreement_adv_3.jsonl',
       'anaphor_agreement_adv_4.jsonl', 'anaphor_agreement_adv_5.jsonl',
       'anaphor_agreement_adv_6.jsonl', 'anaphor_agreement_adv_7.jsonl',
       'anaphor_agreement_basic.jsonl', 
       'anaphor_agreement_rc.jsonl',
       'anaphor_agreement_rc_adj_1.jsonl',
       'anaphor_agreement_rc_adj_2.jsonl',
       'anaphor_agreement_rc_adj_3.jsonl',
       'anaphor_agreement_rc_adj_4.jsonl',
       'anaphor_agreement_rc_adj_5.jsonl', 
       'anaphor_agreement_rn.jsonl',
       'anaphor_agreement_rn_adj_1.jsonl',
       'anaphor_agreement_rn_adj_2.jsonl',
       'anaphor_agreement_rn_adj_3.jsonl',
       'anaphor_agreement_rn_adj_4.jsonl',
       'anaphor_agreement_rn_adj_5.jsonl', 
       
       'sv_agreement_1_adj_1.jsonl',
       'sv_agreement_1_adj_2.jsonl', 'sv_agreement_1_adj_3.jsonl',
       'sv_agreement_1_adj_4.jsonl', 'sv_agreement_1_adj_5.jsonl',
       'sv_agreement_1_adj_6.jsonl', 'sv_agreement_1_adj_7.jsonl',
       'sv_agreement_1_adv_1.jsonl', 
       'sv_agreement_1_adv_2.jsonl',
       'sv_agreement_1_adv_3.jsonl', 'sv_agreement_1_adv_4.jsonl',
       'sv_agreement_1_adv_5.jsonl', 'sv_agreement_1_adv_6.jsonl',
       'sv_agreement_1_adv_7.jsonl', 
       'sv_agreement_1_basic.jsonl',
       'sv_agreement_1_rc.jsonl', 'sv_agreement_1_rc_adj_1.jsonl',
       'sv_agreement_1_rc_adj_2.jsonl', 'sv_agreement_1_rc_adj_3.jsonl',
       'sv_agreement_1_rc_adj_4.jsonl', 'sv_agreement_1_rc_adj_5.jsonl',
       'sv_agreement_1_rn.jsonl', 
       'sv_agreement_1_rn_adj_1.jsonl',
       'sv_agreement_1_rn_adj_2.jsonl', 'sv_agreement_1_rn_adj_3.jsonl',
       'sv_agreement_1_rn_adj_4.jsonl', 'sv_agreement_1_rn_adj_5.jsonl',

       'wh_island_adj_1.jsonl', 'wh_island_adj_2.jsonl',
       'wh_island_adj_3.jsonl', 'wh_island_adj_4.jsonl',
       'wh_island_adj_5.jsonl', 'wh_island_adj_6.jsonl',
       'wh_island_adj_7.jsonl', 
       'wh_island_adv_1.jsonl',
       'wh_island_adv_2.jsonl', 'wh_island_adv_3.jsonl',
       'wh_island_adv_4.jsonl', 'wh_island_adv_5.jsonl',
       'wh_island_adv_6.jsonl', 'wh_island_adv_7.jsonl',
       'wh_island_basic.jsonl', 
       'wh_island_rc.jsonl',
       'wh_island_rc_adj_1.jsonl', 'wh_island_rc_adj_2.jsonl',
       'wh_island_rc_adj_3.jsonl', 'wh_island_rc_adj_4.jsonl',
       'wh_island_rc_adj_5.jsonl', 
       'wh_island_rn.jsonl',
       'wh_island_rn_adj_1.jsonl', 'wh_island_rn_adj_2.jsonl',
       'wh_island_rn_adj_3.jsonl', 'wh_island_rn_adj_4.jsonl',
       'wh_island_rn_adj_5.jsonl']
}


def accuracy_on_task(task_name, eval_model, template_name, num_fewshot):


    # predictions_path = os.path.join(args.model_path, "zeroshot", task_title, "predictions.txt")
    # predictions_dir = os.path.dirname(predictions_path)

    if args.output_dir != ".":
        predictions_path = os.path.join(args.output_dir, "zeroshot", task_title, "predictions.txt")
        predictions_dir = os.path.dirname(predictions_path)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
    else:
        predictions_path = os.path.join(args.model_path, "zeroshot", task_title, "predictions.txt")
        predictions_dir = os.path.dirname(predictions_path)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)

    # if not os.path.exists(predictions_dir):
    #     os.makedirs(predictions_dir)
    
    print(f"Loading task: {task}")
    eval_task = lm_eval.get_task_list(task_name, template_names=[template_name])
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12,
                               num_fewshot=num_fewshot, predictions_path=predictions_path)
    accuracy = results['results'][0]['acc']
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, # added
                        default=None, help="Name of the model to load from HuggingFace Hub")
    parser.add_argument("--model_path", type=str,
                        help="Path to huggingface model and tokenizer.")
    parser.add_argument("--model_type", type=str,
                        choices=["decoder only", "decoder", "encoder only",
                                 "encoder", "encoder-decoder","nanogpt", "nanogpt-dev-fm" ],
                        help="Language model architecture.")

    parser.add_argument("--output_dir", "-o", type=str, default=".",
                        help="Output directory for evaluation results.")
    parser.add_argument("--data_dir", "-d", type=str, default=".",
                        help="Directory containing data and tokenizer")

    parser.add_argument("--tasks", "-t", type=str,
                        choices=["blimp", "supplement", "blimp_test", "all", "dep_dist", "dep_dist_test"],
                        default="all",
                        help="Tasks on which we evaluate.")
    parser.add_argument("--run_aoa", "-a", action="store_true",
                        help="Will run the additional AoA prediction task.")
    parser.add_argument("--trust_remote_code", "-r", action="store_true",
                        help="Trust remote code (e.g. from huggingface) when loading model.")
    parser.add_argument("--num_fewshot", "-n", type=int, default=0,
                        help="Number of few-shot examples to show the model for each test example.")

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="ckpt.pt",
        help="Checkpoint file inside output_dir."
        )   
    args = parser.parse_args()

   

    MODEL_TYPE_REMAP = {"decoder only": "hf-causal", "decoder": "hf-causal",
                        "encoder only": "hf-mlm", "encoder": "hf-mlm",
                        "encoder-decoder": "hf-seq2seq",
                        "nanogpt": "nanogpt", "nanogpt-dev-fm": "nanogpt-dev-fm"}

    # if args.model_type == "nanogpt":
    if args.model_type in ["nanogpt", "nanogpt-dev-fm"]:
        eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                       out_dir = args.output_dir,
                                       data_dir = args.data_dir,
                                       checkpoint_name=args.checkpoint_name,
                                       model_name = args.model_name, # added
                                       device="cuda")
    else:
        eval_model = lm_eval.get_model(MODEL_TYPE_REMAP[args.model_type],
                                       pretrained=args.model_path,
                                       trust_remote_code=args.trust_remote_code,
                                       device="cuda")
    tasks = []
    if args.tasks == "all":
        for task_type in TASKS.keys():
            if task_type != "blimp_test":
                tasks.extend(TASKS[task_type])
    else:
        tasks = TASKS[args.tasks]

    accuracies = {}
    # Iterate through tasks, get accuracies
    for task in tasks:
        if task in TASKS["blimp"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/blimp_filtered/{task}"
        elif task in TASKS["supplement"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/supplement_filtered/{task}"
        elif task in TASKS["dep_dist"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/blimp_dep_dist/{task}"
        elif task in TASKS["dep_dist_test"]:
            template = None
            task_title = task.split(".json")[0]
            task = f"blimp_from_file:filter-data/blimp_dep_dist/{task}"
        else:
            raise ValueError("Unrecognized task!")
        accuracies[task_title] = accuracy_on_task(task, eval_model, template,
                    args.num_fewshot)
        print(f"{task_title}:\t{accuracies[task_title] * 100:.2f}%")

        # Write scores to file
        if args.output_dir != ".":
            out_path = os.path.join(args.output_dir, "zeroshot", task_title, "eval_results.json")
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_path = os.path.join(args.model_path, "zeroshot", task_title, "eval_results.json")
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        #out_path = os.path.join(args.model_path, "zeroshot", task_title, "eval_results.json")
        with open(out_path, 'w') as out_file:
            json.dump({"eval_accuracy": accuracies[task_title]}, out_file)

        # out_path = os.path.join(args.model_path, "zeroshot", task_title, "eval_results.json")
        # out_dir = os.path.dirname(out_path)
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)

    # Print scores
    print("\nScores:")
    for task in accuracies.keys():
        print(f"{task}:\t{accuracies[task] * 100:.2f}%")


    if args.run_aoa:
        # Run AoA prediction evaluation
        word_surprisals_n, mad_results = lm_eval.aoa_pred_eval(eval_model.model, eval_model.tokenizer, MODEL_TYPE_REMAP[args.model_type], batch_size = 32)
        out_dir = os.path.join(args.model_path, "aoa_prediction")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "extracted_average_surprisals.json") , 'w') as out_file:
            json.dump(word_surprisals_n, out_file)
        with open(os.path.join(out_dir, "mean_absolute_deviation_results.json"), 'w') as out_file:
            json.dump(mad_results, out_file)


#python babylm_eval.py --model_type=nanogpt --output_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-2x2-nomask-1709513019 --data_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data/babylm_full_bpe --tasks  blimp_test
#python babylm_eval.py --model_type=nanogpt --output_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump/test_gpt2_eval --tasks  blimp_test
#python babylm_eval.py --model_type=nanogpt --output_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump/test_GPT_wee_nanogpt_eval --data_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data/babylm_full_bpe_8k --tasks  blimp_test
#python babylm_eval.py --model_type=nanogpt --output_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump/test_GPT_wee_nanogpt_eval --data_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data/GPT_wee_hf --tasks  blimp_test

#python babylm_eval.py --model_type=nanogpt --output_dir /home/fleetprobe/fleetprobe/output_dump/test_gpt2_eval --tasks  blimp_test