from transformers import AutoModelForImageClassification

efficientnet_model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b7")
efficientnet_model.save_pretrained("efficientnet-b7")