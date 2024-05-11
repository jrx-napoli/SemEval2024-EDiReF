import sys

from dataset_generation import get_dataloaders
from bert_dataset_generation import get_dataloaders as get_dataloaders_bert
from evaluate import evaluate, evaluate_bert
from models import LSTM
from bert_model import BERT
from options import get_args
from setup import setup_devices
from train import train, train_bert


def run(args):
    device = setup_devices(args)
    train_dataloader, val_dataloader = get_dataloaders(args)
    vocab_size = train_dataloader.dataset.vocab_size
    distinct_labels_count = train_dataloader.dataset.distinct_labels_count
    model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=256, output_dim=distinct_labels_count).to(device)
    model = train(model=model, train_dataloader=train_dataloader, n_epochs=args.n_epochs, device=device)
    evaluate(model=model, test_dataloader=val_dataloader, device=device)


def run_bert(args):
    device = setup_devices(args)
    train_dataloader, val_dataloader = get_dataloaders_bert(args)
    model = BERT().to(device)
    model = train_bert(model=model, train_dataloader=train_dataloader, n_epochs=args.n_epochs, device=device)
    evaluate_bert(model=model, data_loader=val_dataloader, device=device)


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    if args.experiment_name == "erc":
        run(args)
    elif args.experiment_name == "erc_bert":
        run_bert(args)
