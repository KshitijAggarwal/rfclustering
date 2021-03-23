#!/usr/bin/env python
# coding: utf-8

from astropy import time
from rfpipe import source, metadata, util, pipeline, state
import numpy as np
import logging, os, argparse, glob


def inject_one(preffile, devicenum, outdir):
    """
    Script to inject one simulated FRB on simulated data and save unclustered candidates.

    :param preffile: Preference file with search preferences
    :param devicenum: GPU devicenumber
    :param outdir: Output directory
    :return:
    """
    configs = ["A", "B", "C", "D"]
    bands = ["L", "S", "X", "C"]
    config = configs[np.random.randint(len(configs))]
    band = bands[np.random.randint(len(bands))]

    t0 = time.Time.now().mjd
    meta = metadata.mock_metadata(
        t0,
        t0 + 10 / (24 * 3600),
        20,
        11,
        32 * 4 * 2,
        2,
        5e3,
        scan=1,
        datasource="sim",
        antconfig=config,
        band=band,
    )

    dataset = meta["datasetId"] + "_config_" + config + "_band_" + band

    workdir = outdir + "/" + dataset

    try:
        os.mkdir(workdir)
    except FileExistsError:
        logging.info("Directory {0} exists, using it.".format(workdir))
    except OSError:
        logging.info("Can't create directory {0}".format(workdir))
    else:
        logging.info("Created directory {0}".format(workdir))

    prefs = {}
    prefs["workdir"] = workdir
    prefs["savenoise"] = False
    prefs["fftmode"] = "fftw"
    prefs["nthread"] = 10
    prefs["flaglist"] = [
        ("badchtslide", 4.0, 20),
        ("badchtslide", 4, 20),
        ("badspw", 4.0),
        ("blstd", 3.5, 0.008),
    ]

    st = state.State(
        inmeta=meta,
        showsummary=False,
        preffile=preffile,
        name="NRAOdefault" + band,
        inprefs=prefs,
    )
    segment = 0
    data = source.read_segment(st, segment)
    dmind = None
    dtind = None
    snr = np.random.uniform(low=10, high=40)
    mock = util.make_transient_params(
        st, snr=snr, segment=segment, data=data, ntr=1, lm=-1, dmind=dmind, dtind=dtind
    )

    st.clearcache()
    st.prefs.simulated_transient = mock

    cc = pipeline.pipeline_seg(st=st, segment=segment, devicenum=devicenum)

    if not len(cc):
        logging.info(
            "No candidate found. Deleting the empty pickle, and trying with a higher SNR now."
        )
        pkl = glob.glob(cc.state.prefs.workdir + "/*pkl")[0]
        try:
            os.remove(pkl)
        except OSError as e:
            pass
        snr = snr + 5
        mock = util.make_transient_params(
            st,
            snr=snr,
            segment=segment,
            data=data,
            ntr=1,
            lm=-1,
            dmind=dmind,
            dtind=dtind,
        )

        st.clearcache()
        st.prefs.simulated_transient = mock
        cc = pipeline.pipeline_seg(st=st, segment=segment, devicenum=devicenum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject simulated FRB on simulated data and save"
        "unclustered candidates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--devicenum", help="GPU numbers", required=False, type=int, default=0
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
        default="/hyrule/data/users/kshitij/hdbscan/final_data/clean/",
    )

    values = parser.parse_args()

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        logging.info("Argument %s: %r", arg, value)

    inject_one(
        preffile=values.preffile, devicenum=str(values.devicenum), outdir=values.outdir
    )
