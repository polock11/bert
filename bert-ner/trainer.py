import torch
from sklearn.metrics import accuracy_score

class Trainer:

    def __init__(self, model, training_loader, optimizer, DEVICE, MAX_GRAD_NORM):
        self.model = model
        self.training_loader = training_loader
        self.device = DEVICE
        self.max_grad_norm = MAX_GRAD_NORM
        self.optimizer = optimizer

    def train(self, epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.model.train()
        
        for idx, batch in enumerate(self.training_loader):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            labels = batch['labels'].to(self.device, dtype=torch.long)

            # Forward pass
            outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            tr_logits = outputs.logits
            
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)
            
            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_labels.extend(labels.cpu().numpy())
            tr_preds.extend(predictions.cpu().numpy())

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.max_grad_norm
            )
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def fit(self, epochs):
        for ep in range(epochs):
            print(f"Training epoch: {ep + 1}")
            self.train(ep)