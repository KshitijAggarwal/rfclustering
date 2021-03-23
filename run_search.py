# coding: utf-8
# !/usr/bin/env python

import argparse
import glob
import logging
import os

logger = logging.getLogger()
from rfpipe.pipeline import pipeline_sdm


def process(sdm, gainpath, preffile, devicenum, outdir):
    """
    Run rfpipe on an SDM and save unclustered results.

    :param sdm: SDM file
    :param gainpath: path of gainfile
    :param preffile: preference file
    :param devicenum: GPU number
    :param outdir: output directory
    :return:
    """
    sdmname = sdm
    if sdmname[-1] == "/":
        sdmname = sdmname[:-1]

    if os.path.basename(sdmname).split("_")[0] == "realfast":
        datasetId = "{0}".format("_".join(os.path.basename(sdmname).split("_")[1:-1]))
    else:
        datasetId = os.path.basename(sdmname)

    datasetId = ".".join(
        [x for x in datasetId.split(".") if "scan" not in x and "cut" not in x]
    )

    datadir = outdir + "/" + os.path.basename(sdmname)

    try:
        os.mkdir(datadir)
    except FileExistsError:
        logging.info("Directory {0} exists, using it.".format(datadir))
    except OSError:
        logging.info("Can't create directory {0}".format(datadir))
    else:
        logging.info("Created directory {0}".format(datadir))

    gainname = datasetId + ".GN"
    logging.info("Searching for the gainfile {0} in {1}".format(gainname, gainpath))
    gainfile = []
    for path, dirs, files in os.walk(gainpath):
        for f in filter(lambda x: gainname in x, files):
            gainfile = os.path.join(path, gainname)
            break
    try:
        assert len(gainfile)
        logging.info("Found gainfile for {0} in {1}".format(datasetId, gainfile))
    except AssertionError as err:
        logging.error("No gainfile found for {0} in {1}".format(datasetId, gainfile))
        raise err

    prefs = {}
    prefs["workdir"] = datadir
    prefs["gainfile"] = gainfile
    prefs["savenoise"] = False
    prefs["devicenum"] = devicenum

    prefs["flaglist"] = [
        ("badchtslide", 4.0, 20),
        ("badchtslide", 4, 20),
        ("badspw", 4.0),
        ("blstd", 3.5, 0.008),
    ]

    pipeline_sdm(
        sdm, inprefs=prefs, intent="TARGET", preffile=preffile, devicenum=devicenum
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normal pipeline search on an SDM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-f", "--file", help="Path of SDM", required=True, type=str)
    parser.add_argument(
        "-d", "--devicenum", help="GPU numbers", required=False, type=int, default=0
    )
    parser.add_argument(
        "-g",
        "--gainpath",
        help="Path of gain files",
        required=False,
        type=str,
        default="/hyrule/data/users/kshitij/hdbscan/gainfiles/",
    )
    parser.add_argument(
        "-p",
        "--preffile",
        help="Path of preffile",
        required=False,
        type=str,
        default="/hyrule/data/users/kshitij/hdbscan/scripts/final/realfast_nocluster.yml",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory",
        required=False,
        type=str,
        default="/hyrule/data/users/kshitij/hdbscan/final_data/",
    )

    values = parser.parse_args()

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        logging.info("Argument %s: %r", arg, value)

    sdm = glob.glob(values.file)[0]

    process(
        sdm=sdm,
        gainpath=values.gainpath,
        preffile=values.preffile,
        devicenum=str(values.devicenum),
        outdir=values.outdir,
    )
