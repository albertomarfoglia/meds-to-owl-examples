import argparse
from pathlib import Path

from generation.sphn_generation import gen_sphn_kg
from generation.preprocess_data_sphn import preprocess_kg
from models.node_pred_rgcn_sphn import run_rgcn


parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=10000, help='number of patients')
parser.add_argument('--timeOpt', type=str, default='TS', choices=['NT', 'TS', 'TR', 'TS_TR'], help='time information option')
parser.add_argument('--folds', type=int, default=10, help='number of folds for CV')
parser.add_argument('--dr', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension of the model')
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    # gen_sphn_kg(args.num_patients, args.timeOpt, data_path=Path(f"{PROJECT_ROOT}/data/syn_data_{args.num_patients}.csv"))
    preprocess_kg(
        num_patients=args.num_patients,
        time_opt=args.timeOpt,
        input_dir=Path(f"{PROJECT_ROOT}/data"),
        output_dir=Path(f"{PROJECT_ROOT}/processed_data"),
        file_prefix="meds")
    
    run_rgcn(
        args.num_patients, args.folds, args.timeOpt, 
        dr=args.dr, 
        lr=args.lr, 
        wd=args.wd, 
        embed_dim=args.embed_dim, 
        hidden_dim=args.hidden_dim,
        prefix="meds",
        root = PROJECT_ROOT
    )
    # print("Model training and evaluation completed.")