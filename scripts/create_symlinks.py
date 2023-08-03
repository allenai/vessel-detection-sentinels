import argparse
import os


def create_links(data_dir: str) -> None:
    """Create symbolic links for overlaps in Sentinel-1 training data.

    Parameters
    ----------
    data_dir: str
        Absolute path to local directory containing: 1) Preprocessed Sentinel-1 image
        folders, and 2) A file `overlap_links.txt` describing one symlink per line,
        in the format '(data_dir relative path to symlink source) (data_dir relative path to symlink target)'.

    """
    link_file = os.path.join(data_dir, 'overlap_links.txt')
    with open(link_file, 'r') as f:
        tot = sum(1 for line in f)
        f.seek(0)
        created = 0
        skipped = 0
        log_freq = 100000
        for line in f:
            line = line.strip()
            src, tgt = line.split(" ")

            # Get base paths
            src = "/".join(src.split("/")[1:])
            tgt = "/".join(tgt.split("/")[2:])

            # Get absolute dir path
            src_dir = "/".join(src.split("/")[:-1])
            src_dir = os.path.join(data_dir, src_dir)

            # Make dirs as necessary
            os.makedirs(src_dir, exist_ok=True)

            # Get absolute src, tgt
            tgt = os.path.join(data_dir, tgt)
            src = os.path.join(data_dir, src)
            try:
                os.symlink(tgt, src)
                if created % log_freq == 0:
                    print(f"Created {created}/{tot} links.")
                created += 1
            except FileExistsError:
                print(f"Skipping existing link: {src}")
                skipped += 1
                pass
            except Exception as e:
                print(f"Unexpected exception: {e}")
        print(f"Created {created} symlinks.")
        print(f"Skipped {skipped} existing symlinks.")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="./preprocess",
                        help='Absolute path to preprocess directory.')
    args = parser.parse_args()
    args_dict = vars(args)
    create_links(**args_dict)
