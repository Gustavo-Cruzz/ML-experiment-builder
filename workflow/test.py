import mlflow


def test_routine(params, experiment_id):
    print("\n\ntesting")
    save_path = f"..{params['save_path']}/{experiment_id}/test/"

    # List all runs
    runs = mlflow.search_runs()

    # Filter runs based on parameters
    filtered_runs = mlflow.search_runs(
        filter_string="params.param_name = 'param_value'"
    )

    # Access a specific run
    run = mlflow.search_runs(run_id=save_path).iloc[0]
