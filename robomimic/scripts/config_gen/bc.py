from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    if args.env == "robocasa":
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                #(get_ds_cfg("robocasa_round0", src="human", eval=EVAL_TASKS), "robocasa-bc"),
                #(get_ds_cfg("robocasa_round01", src="human", eval=EVAL_TASKS), "robocasa-bc"),
                (get_ds_cfg("robocasa_round012", src="human", eval=EVAL_TASKS), "robocasa-bc"),
            ]
        )
    elif args.env == "mutex":
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=20000,
            values_and_names=[
                (get_ds_cfg("mutex_round0", src="mutex", eval=EVAL_TASKS, overwrite_ds_lang=True), "mutex-bc"),
            ]
        )

        generator.add_param(
            key="train.data_format",
            name="",
            group=2,
            values=["mutex"],
        )


    # train for 2k epochs and save every 100 epochs to be safe
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2000],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[100],
    )

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[6],
    )

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    # finetune 
    if args.ckpt_path is not None:

        generator.add_param(
            key="experiment.ckpt_path",
            name="finetuned",
            group=-1,
            values=[args.ckpt_path],
            value_names=["T"],
        )

        generator.add_param(
            key="algo.optim_params.policy.learning_rate.scheduler_type",
            name="sche",
            group=-1,            
            values=["constant"],
            value_names=["const"],
        )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
    )
    
    args = parser.parse_args()
    
    make_generator(args, make_generator_helper)
