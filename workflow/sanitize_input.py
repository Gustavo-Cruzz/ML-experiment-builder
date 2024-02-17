class Validate:
    def __init__(self, parameters):
        self.parameters = parameters
        self.train = parameters.get("train")

        if self.train is None:
            raise "Please, provide train parameters"

        self.validade_train()

    def assert_variable_type(self, variable, expected_type):
        """
        Asserts whether a variable is of a certain type.

        Args:
            variable: The variable to check.
            expected_type: The expected type of the variable.

        Raises:
            AssertionError: If the variable is not of the expected type.
        """
        if not isinstance(variable, expected_type):
            raise AssertionError(f"Variable is not of type {expected_type}")

    def validade_train(self):
        """
        Validates training parameters defined in the YAML file.

        Raises:
            ValueError: If any training parameter is invalid or missing.
        """
        valid_models = ["ResNet50", "VGG16", "mobile_netv2"]
        activation_funcs = ["softmax", "sigmoid", "relu"]
        loss_funcs = [
            "binary_crossentropy",
            "binary_focal_crossentropy",
            "categorical_crossentropy",
            "categorical_focal_crossentropy",
            "sparse_categorical_crossentropy",
        ]

        checks = {
            "model_name": valid_models,
            "loss_func": loss_funcs,
            "activation_func": activation_funcs,
        }

        for i in checks.keys():
            if self.train.get(i) not in checks[i]:
                raise ValueError(
                    f"""{self.train.get(i)} is not a valid {i} parameter
              Valid {i}s are {checks[i]}"""
                )

        other_params = {
            "classes": int,
            "save_path": str,
            "batch_size": int,
            "epochs": int,
            "dataset_name": str,
            "skip": bool,
        }

        for item in other_params.keys():
            if item not in self.train.keys():
                raise ValueError(f"""Parameter {item} is missing in train YAML""")

            self.assert_variable_type(self.train[item], other_params[item])

        if not self.train.get("image_size"):
            raise ValueError("""Parameter image_size is invalid in YAML""")

        if len(self.train["image_size"]) < 3:
            raise ValueError(
                """Parameter image_size should be passed like [height, width, channels]"""
            )
