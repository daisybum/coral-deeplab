__version__ = "0.4"

from . import pretrained
from ._downloads import from_precompiled


try:
    from . import applications
    from . import layers
    from . import attention
    from . import fusion

except ImportError:
    print(
        "coral-deeplab is running without tensorflow dependencies. "
        "You can still use it to pull precompiled models."
    )
