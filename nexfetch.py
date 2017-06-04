import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import pyart
from pyart.io.nexrad_level3 import NEXRADLevel3File as N3F
from metpy.io import Level3File
import numpy as np
import multiprocessing as mp
import os, sys
from pymongo import MongoClient
from bson.binary import Binary
from datetime import datetime, timedelta

XS = 2.
SM = 5.
MD = 10.
LG = 20.

SIZES = [XS, SM, MD, LG]

RNG = 230.

CODE_CORRELATION = {
    19: {
        0.5: {
            'code': 'N0R',
            'title': 'Base Reflectivity - Tilt 1',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        }
    },
    27: {
        0.5: {
            'code': 'N0V',
            'title': 'Base Velocity - Tilt 1',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        }
    },
    30: {
        0.5: {
            'code': 'NSW',
            'title': 'Base Spectrum Width - Tilt 1',
            'color': 'pyart_NWS_SPW',
            'type': 'spectrum_width'
        }
    },
    56: {
        0.5: {
            'code': 'N0S',
            'title': 'Storm Relative Mean Radial Velocity - Tilt 1',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        1.3: {
            'code': 'N1S',
            'title': 'Storm Relative Mean Radial Velocity - Tilt 2',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        2.4: {
            'code': 'N2S',
            'title': 'Storm Relative Mean Radial Velocity - Tilt 3',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        3.1: {
            'code': 'N3S',
            'title': 'Storm Relative Mean Radial Velocity - Tilt 4',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        }
    },
    94: {
        0.5: {
            'code': 'N0Q',
            'title': 'Base Reflectivity Data Array - Tilt 1',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        },
        0.9: {
            'code': 'NAQ',
            'title': 'Base Reflectivity Data Array - Tilt 2',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        },
        1.5: {
            'code': 'N1Q',
            'title': 'Base Reflectivity Data Array - Tilt 3',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        },
        1.8: {
            'code': 'NBQ',
            'title': 'Base Reflectivity Data Array - Tilt 4',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        },
        2.4: {
            'code': 'N2Q',
            'title': 'Base Reflectivity Data Array - Tilt 5',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        },
        3.1: {
            'code': 'N3Q',
            'title': 'Base Reflectivity Data Array - Tilt 6',
            'color': 'pyart_NWSRef',
            'type': 'reflectivity'
        }
    },
    99: {
        0.5: {
            'code': 'N0U',
            'title': 'Base Velocity Data Array - Tilt 1',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        0.9: {
            'code': 'NAU',
            'title': 'Base Velocity Data Array - Tilt 2',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        1.3: {
            'code': 'N1U',
            'title': 'Base Velocity Data Array - Tilt 3',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        1.8: {
            'code': 'NBU',
            'title': 'Base Velocity Data Array - Tilt 4',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        2.4: {
            'code': 'N2U',
            'title': 'Base Velocity Data Array - Tilt 5',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        },
        3.1: {
            'code': 'N3U',
            'title': 'Base Velocity Data Array - Tilt 6',
            'color': 'pyart_NWSVel',
            'type': 'velocity'
        }
    },
    161: {
        0.5: {
            'code': 'N0C',
            'title': 'Digital Correlation Coefficient - Tilt 1',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        },
        0.9: {
            'code': 'NAC',
            'title': 'Digital Correlation Coefficient - Tilt 2',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        },
        1.3: {
            'code': 'N1C',
            'title': 'Digital Correlation Coefficient - Tilt 3',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        },
        1.8: {
            'code': 'NBC',
            'title': 'Digital Correlation Coefficient - Tilt 4',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        },
        2.4: {
            'code': 'N2C',
            'title': 'Digital Correlation Coefficient - Tilt 5',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        },
        3.1: {
            'code': 'N3C',
            'title': 'Digital Correlation Coefficient - Tilt 6',
            'color': 'pyart_RefDiff',
            'type': 'cross_correlation_ratio'
        }
    }
}

def totimestamp(dt, epoch=datetime(1970, 1, 1)):
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 1e6

def neighborhood(iterable):
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)

def process(i, j, item1, next1, item2, next2):
    if next1 != None and next2 != None:
        RAD.set_limits(xlim=(item1, next1), ylim=(item2, next2), ax=ax)
        fig.set_size_inches(siz, siz)
        try:
            client = MongoClient()
            db = client['radars']
            fig.savefig('pyfile_' + str(int(siz)) + '_' + str(i) + '_' + str(j) + '.png', transparent=True)
            col = db['K' + l3rad.siteID]
            doc = {
                "product": rad,
                "zoom": str(int(siz)),
                "timestamp": str(int(totimestamp(N3F(radarFile).get_volume_start_datetime()))),
                "tile": Binary(open('pyfile_' + str(int(siz)) + '_' + str(i) + '_' + str(j) + '.png', 'rb').read()),
                "tileName": str(i) + '_' + str(j),
                "code": str(code),
                "angle": str(angle)
            }

            doc_id = col.insert_one(doc).inserted_id
            print(doc_id)
            os.remove('pyfile_' + str(int(siz)) + '_' + str(i) + '_' + str(j) + '.png')
        except OSError:
            pass
    return

if __name__ == '__main__':
    for siz in SIZES:
        radarFile = sys.argv[1]
        radarRAD = pyart.io.read_nexrad_level3(radarFile)
        RAD = pyart.graph.RadarDisplay(radarRAD)
        l3rad = Level3File(radarFile)
        code = l3rad.header.code
        angle = round(l3rad.metadata['el_angle'], 1)
        
        fig = plt.figure(figsize=(siz, siz))

        rad = CODE_CORRELATION[code][angle]['code']

        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        RAD.plot(CODE_CORRELATION[code][angle]['type'], ax=ax, title_flag=False, colorbar_flag=False, axislabels_flag=False, cmap=CODE_CORRELATION[code][angle]['color'])
        xmax = np.arange(0, RNG, 44.5)
        ymax = np.arange(0, RNG, 55.5)
        xmin = -xmax
        ymin = -ymax
        xtotran = np.concatenate([xmin, xmax])
        ytotran = np.concatenate([ymin, ymax])
        xnew = np.unique(np.sort(xtotran))
        ynew = np.unique(np.sort(ytotran))

        jobs = []
        lims = []
        i = 0
        for prev1, item1, next1 in neighborhood(xnew):
            i += 1
            j = 0
            for prev2, item2, next2 in neighborhood(ynew):
                j += 1
                lims.append((i, j, item1, next1, item2, next2))
        for z, y, a, b, c, d in lims:
            p = mp.Process(target=process, args=(z, y, a, b, c, d))
            jobs.append(p)
            p.start()