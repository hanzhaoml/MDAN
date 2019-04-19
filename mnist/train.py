import argparse
import sys

import net_flow
import read_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='config')
    args = parser.parse_args()
    model_params = read_config.read(args.config_file_path)
    print(model_params)

    net = net_flow.NetFlow(model_params, True, True)
    net.mainloop()
