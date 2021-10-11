import logging
import os, sys
import argparse

import emmental
from emmental import Meta
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config


from emmental_utils import (
    fetch_dataloaders,
    create_tasks,
    write_to_file,
    save_predictions,
)
from utils import add_application_args

logger = logging.getLogger(__name__)


def get_parser():
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Observational Runner", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)
    return parser


def main(args):

    # Ensure that global state is fresh
    Meta.reset()

    # Initialize Emmental
    config = parse_args_to_config(args)
    emmental.init(
        config["meta_config"]["log_path"],
        config=config,
        use_exact_log_path=config["meta_config"]["use_exact_log_path"],
    )

    # Save command line argument into file
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(Meta.log_path, "cmd.txt", cmd_msg)

    # Save Emmental config into file
    logger.info(f"Config: {Meta.config}")
    write_to_file(Meta.log_path, "config.txt", Meta.config)

    # fetch dataloaders
    dataloaders = fetch_dataloaders(
        args.task,
        args.gaze_mtl_task,
        args.source,
        args.data_dir,
        args.train_scale,
        args.val_scale,
        args.seed,
        args.batch_size,
    )

    # create emmental tasks
    tasks = create_tasks(
        args.task,
        args.gaze_mtl_task,
        args.source,
        args.pretrained,
        args.task_weights,
        load_path=args.load_cnn,
    )

    # create emmental model
    model = EmmentalModel(name="Observational", tasks=tasks)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Model task weights: {model.weights}")

    if not (args.evaluate):
        # run learner
        emmental_learner = EmmentalLearner()

        if args.transfer_learning:
            model.weights = {"target": 1.0, "helper_task_0": 0}

        emmental_learner.learn(model, dataloaders)

        if args.transfer_learning:
            model.weights = {"target": 0.0, "helper_task_0": 1.0}
            # Freeze CNN encoder
            cnn_encoder = model.module_pool.cnn
            for param in cnn_encoder.parameters():
                param.requires_grad = False

            Meta.config["learner_config"]["n_epochs"] = 15
            emmental_learner.learn(model, dataloaders)

        if args.load_cnn:
            cnn_encoder = model.module_pool.cnn
            for param in cnn_encoder.parameters():
                param.requires_grad = False

        scores = model.score(dataloaders)

        # Save metrics into file
        write_to_file(Meta.log_path, "metrics.txt", scores)
        save_predictions(Meta.log_path, model, dataloaders, args.task)

        # Save best metrics into file
        write_to_file(
            Meta.log_path,
            "best_metrics.txt",
            emmental_learner.logging_manager.checkpointer.best_metric_dict,
        )

        # save_features(Meta.config["meta_config"]["log_path"],model,dataloaders)

    else:

        best_model_pth = os.path.join(
            Meta.config["meta_config"]["log_path"],
            "best_model_target_" + args.source + "_val_accuracy.pth",
        )
        model.load(best_model_pth)

        # Save metrics into file
        # write_to_file(Meta.log_path, "metrics.txt", scores)
        save_predictions(Meta.log_path, model, dataloaders, args.task)
        # save_features(Meta.config["meta_config"]["log_path"],model,dataloaders)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
