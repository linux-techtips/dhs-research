import lib as depth

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth estimation')
    parser.add_argument(
        'image',
        metavar='image',
        type=str,
        help='input image path',
    )
    parser.add_argument(
        '--model',
        type=depth.Model.from_name,
        default='Small',
        choices=list(depth.Model),
        help='depth model to use',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='result.npy',
        help='output file path',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='display the results of the depth generation',
    )

    args = parser.parse_args()

    device, model, transform = depth.init(args.model)
    result = depth.estimate_depth(depth.load_image(args.image), device, model, transform)

    result.dump(args.output)
    depth.show_results(depth.normalize(result))
