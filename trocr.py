import logging
import io
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderConfig  
)
from datasets import Dataset
import pandas as pd

class TrOCR:
    def __init__(
        self,
        processor=None,
        model=None,
        model_name_processor="microsoft/trocr-base-handwritten",
        model_name_encoder_decoder="microsoft/trocr-base-handwritten",
        cache_dir=None,
        force_download=False,
        revision="main",
        model_config_overrides={},
        log_level=logging.INFO
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Initialize processor
        if processor is not None:
            self.processor = processor
        else:
            self.processor = TrOCRProcessor.from_pretrained(
                model_name_processor,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision
            )
            self.logger.debug("Initialized processor from pretrained model.")

        # Initialize model
        if model is not None:
            self.model = model
        else:
            if model_config_overrides:
                # Create model config with overrides
                config = VisionEncoderDecoderConfig.from_pretrained(
                    model_name_encoder_decoder,
                    **model_config_overrides
                )

                # Ensure that the decoder_start_token_id is set
                if config.decoder_start_token_id is None:
                    # Try to get it from the tokenizer or set it to a default value
                    config.decoder_start_token_id = self.processor.tokenizer.cls_token_id or self.processor.tokenizer.bos_token_id or 0

                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_name_encoder_decoder,
                    config=config,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision
                )
                self.logger.debug("Initialized model with custom configuration.")
            else:
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_name_encoder_decoder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    revision=revision
                )
                self.logger.debug("Initialized model from pretrained model.")

        # Ensure decoder_start_token_id is set
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id or self.processor.tokenizer.bos_token_id or 0

    def process_image(self, image_bytes, image_processing_parameters={}):
        """Converts image bytes to pixel values suitable for the model"""
        image = Image.open(io.BytesIO(image_bytes))
        # Apply additional processing if specified
        if 'convert_mode' in image_processing_parameters:
            image = image.convert(image_processing_parameters['convert_mode'])
        else:
            image = image.convert('RGB')
        if 'resize' in image_processing_parameters:
            image = image.resize(image_processing_parameters['resize'])
        return self.processor(images=image, return_tensors="pt").pixel_values

    def process_text(self, text):
        """Tokenizes input text if available"""
        return self.processor.tokenizer(text, return_tensors="pt").input_ids if text else None

    def generate(self, pixel_values, prediction_parameters={}, **generate_kwargs):
        """Generates text from pixel values using the model"""
        # Move pixel_values to the same device as the model
        pixel_values = pixel_values.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=prediction_parameters.get('max_length', 50),
                num_beams=prediction_parameters.get('num_beams', 4),
                early_stopping=prediction_parameters.get('early_stopping', True),
                **generate_kwargs
            )
        return generated_ids


    def decode(self, generated_ids, prediction_parameters={}):
        """Decodes generated text IDs to strings"""
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=prediction_parameters.get('skip_special_tokens', True),
            clean_up_tokenization_spaces=prediction_parameters.get('clean_up_tokenization_spaces', True)
        )

    def predict(self, image_bytes_list, prediction_parameters={}, image_processing_parameters={}):
        """Processes a list of images and predicts text"""
        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in image_bytes_list]
        # Apply image processing parameters
        processed_images = []
        for image in images:
            if 'convert_mode' in image_processing_parameters:
                image = image.convert(image_processing_parameters['convert_mode'])
            else:
                image = image.convert('RGB')
            if 'resize' in image_processing_parameters:
                image = image.resize(image_processing_parameters['resize'])
            processed_images.append(image)

        pixel_values = self.processor(images=processed_images, return_tensors="pt").pixel_values
        generated_ids = self.generate(pixel_values, prediction_parameters)
        return self.decode(generated_ids, prediction_parameters)

    def test(self, dataset, split='test', batch_size=8, prediction_parameters={}, image_processing_parameters={}):
      if split not in dataset:
          raise ValueError(f"Split '{split}' not found in the dataset.")

      results = []
      dataloader = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
      for batch in dataloader:
          image_bytes_list = batch['image']
          expected_texts = batch['text']
          predicted_texts = self.predict(
              image_bytes_list,
              prediction_parameters=prediction_parameters,
              image_processing_parameters=image_processing_parameters
          )
          for expected_text, predicted_text in zip(expected_texts, predicted_texts):
              results.append({
                  "expected": expected_text,
                  "prediction_0": predicted_text
              })

      return pd.DataFrame(results)


    def finetune(
        self,
        train_dataset,
        eval_dataset=None,
        fine_tuning_parameters={},
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        custom_trainer_class=None,
        freeze_encoder=False,
        freeze_decoder=False,
        logging_steps=500
    ):
        """
        Fine-tunes the model on the provided training dataset.
        """
        if not isinstance(train_dataset, Dataset):
            raise ValueError("Input must be a HuggingFace Dataset containing 'text' and 'image' columns.")

        # Optionally freeze the encoder and/or decoder
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.logger.debug("Encoder layers have been frozen.")
        if freeze_decoder:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            self.logger.debug("Decoder layers have been frozen.")

        # Define the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=fine_tuning_parameters.get('output_dir', './trocr_finetuned'),
            per_device_train_batch_size=fine_tuning_parameters.get('per_device_train_batch_size', 8),
            num_train_epochs=fine_tuning_parameters.get('num_train_epochs', 3),
            learning_rate=fine_tuning_parameters.get('learning_rate', 5e-5),
            logging_dir=fine_tuning_parameters.get('logging_dir', './logs'),
            eval_strategy=fine_tuning_parameters.get('eval_strategy', 'no'),
            save_strategy=fine_tuning_parameters.get('save_strategy', 'no'),
            logging_steps=fine_tuning_parameters.get('logging_steps', logging_steps),
            predict_with_generate=True,
            fp16=fine_tuning_parameters.get('fp16', False),
            bf16=fine_tuning_parameters.get('bf16', False),
            remove_unused_columns=False,  # Ensure all columns are available
            **{k: v for k, v in fine_tuning_parameters.items() if k not in [
                'output_dir', 'per_device_train_batch_size', 'num_train_epochs',
                'learning_rate', 'logging_dir', 'eval_strategy', 'save_strategy',
                'logging_steps', 'fp16', 'bf16', 'remove_unused_columns'
            ]}
        )

        # Use default data collator if none is provided
        if data_collator is None:
            def default_data_collator(batch):
                images = [Image.open(io.BytesIO(item['image'])).convert('RGB') for item in batch]
                texts = [item['text'] for item in batch]
                pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
                labels = self.processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                return {"pixel_values": pixel_values, "labels": labels}
            data_collator = default_data_collator

        # Use custom trainer class if provided
        trainer_class = custom_trainer_class or Seq2SeqTrainer

        # Define the trainer
        trainer = trainer_class(
        model=self.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        )
        # Start training
        trainer.train()

    def save_model(self, save_directory):
      os.makedirs(save_directory, exist_ok=True)
      self.model.save_pretrained(save_directory)
      self.processor.save_pretrained(save_directory)
      print(f"Model and processor saved to {save_directory}")


