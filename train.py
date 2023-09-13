import os
import argparse
import logging
import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import DictConfig
from ruamel.yaml import YAML

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(train_json_path, test_json_path, model_name,num_epochs, model_save_path, checkpoint):
    # --- Config Information ---#
    config_path = './configs/config.yaml'
    if not os.path.exists(config_path):
        logging.info("Downloading default config file...")
        BRANCH = 'main'
        os.makedirs('configs', exist_ok=True)
        os.system(f"wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/${BRANCH}/examples/asr/conf/config.yaml")

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    params['model']['train_ds']['manifest_filepath'] = train_json_path
    params['model']['validation_ds']['manifest_filepath'] = test_json_path

    if model_name:
        import copy
        new_opt = copy.deepcopy(params['model']['optim'])
        new_opt['lr'] = 0.001
        # Transfer Learning
        logging.info(f"Loading pre-trained model: {model_name} for transfer learning...")
        asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
        
        logging.info(f"Changing vocabulary from: {asr_model.decoder.vocabulary}")
        asr_model.change_vocabulary(new_vocabulary=[' ', 'a', 'b','d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', "1", "2", "3", "4", "5", "'"])
        logging.info(f"New vocabulary: {asr_model.decoder.vocabulary}")
        # Use the smaller learning rate we set before
        asr_model.setup_optimization(optim_config=DictConfig(new_opt))

        # Point to the data we'll use for fine-tuning as the training set
        asr_model.setup_training_data(train_data_config=params['model']['train_ds'])

        # Point to the new validation data for fine-tuning
        asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])

        # And now we can create a PyTorch Lightning trainer and call `fit` again.
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=num_epochs)
        trainer.fit(asr_model)


    else:
        # Train from scratch
        logging.info("Training ASR model from scratch...")
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=180)
        asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)
        trainer.fit(asr_model)

    # Save the model
    asr_model.save_to(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # Evaluation
    if checkpoint:
        logging.info("Evaluating the model...")
        params['model']['validation_ds']['batch_size'] = 16
        asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])
        asr_model.cuda()
        asr_model.eval()

        wer_nums = []
        wer_denoms = []

        for test_batch in asr_model.test_dataloader():
            test_batch = [x.cuda() for x in test_batch]
            targets = test_batch[2]
            targets_lengths = test_batch[3]        
            log_probs, encoded_len, greedy_predictions = asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
            asr_model._wer.update(greedy_predictions, targets, targets_lengths)
            _, wer_num, wer_denom = asr_model._wer.compute()
            asr_model._wer.reset()
            wer_nums.append(wer_num.detach().cpu().numpy())
            wer_denoms.append(wer_denom.detach().cpu().numpy())
            del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

        logging.info(f"WER = {sum(wer_nums)/sum(wer_denoms)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an ASR model using NeMo.")
    parser.add_argument("--train_json_path", type=str,  default="./train.json", help="Path to the training JSON file.")
    parser.add_argument("--test_json_path", type=str, default = "./test.json", help="Path to the testing JSON file.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the pre-trained model for transfer learning. If not provided, train from scratch.")
    parser.add_argument("--model_save_path", type=str, default="./asr_model.nemo", help="Path to save the trained ASR model.")
    parser.add_argument("--checkpoint", action="store_true", help="Whether to evaluate the model or not.")
    parser.add_argument("--num_epochs", type=int, default=180, help="Number of epochs to train the model.")

    args = parser.parse_args()
    train_and_evaluate(args.train_json_path, args.test_json_path, args.model_name, args.num_epochs, args.model_save_path, args.checkpoint)
