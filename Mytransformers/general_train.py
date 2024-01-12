
from utils import *
from dataset import *
from transformers import RobertaTokenizerFast, BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup


## Params
num_batch = 32
num_epochs = 10
lr = 0.01

## load data
file_path = '../data/raw/bace.csv'
bace_ds = pd.read_csv(file_path)

##

tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
smiles_list = bace_ds["mol"].tolist()
train_smiles, test_smiles = train_test_split(smiles_list, test_size=0.5, random_state=42)

train_dataset = SmilesDataset(train_smiles, tokenizer)
test_dataset = SmilesDataset(test_smiles, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=num_batch, shuffle=False)

# 2. Define the MolBERT model architecture
config = BertConfig(vocab_size=tokenizer.vocab_size)
model = BertForMaskedLM(config)

# 3. Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_training_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps as warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Add this line

    # 4. Evaluate the model
    model.eval()
    total_loss = 0
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

    avg_val_loss = total_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")


test_smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=C(C=C(C=C1)C(=O)O)O"]
for smiles in test_smiles:
    generated_smiles = generate_smiles(smiles, tokenizer, model)
    print(f"Input SMILES: {smiles}")
    print(f"Generated SMILES: {generated_smiles}")

