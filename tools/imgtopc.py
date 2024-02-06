from lib import depth

import argparse

INTRINSIC = [
    [677.89317334, 0, 468.6357665 ],
    [0, 622.98587888, 433.12794922],
    [0, 0, 1                      ],
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Generation')
    parser.add_argument(
        'image',
        metavar='image',
        type=str,
        help='input input path',
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
        help='display the results of the calibration',
    )

    args = parser.parse_args()

    device, model, transforme = depth.init(args.model)
    result = depth.estimate_depth(depth.load_image(args.image), device, model, transforme)

    pcd = depth.depth_to_point_cloud(result, INTRINSIC)
    depth.dump_point_cloud(pcd, args.output)

    if args.show:
        depth.visualize_point_cloud(pcd)