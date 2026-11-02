.. index:: pair: page; Experimental features
.. _doxid-dev_guide_experimental:

Experimental features
=====================

To test aggressive performance optimizations that might affect accuracy without an impact to regular users oneDNN provides experimental features.

Build-time Controls
~~~~~~~~~~~~~~~~~~~

To enable experimental features the library should be built with a CMake option ``ONEDNN_EXPERIMENTAL=ON``. Each experimental feature has to be individually selected using environment variables.

Experimental features
~~~~~~~~~~~~~~~~~~~~~

=========================================  ===================================================================================================================================================================  
Environment variable                       Desc                                                                                                                                                                 
=========================================  ===================================================================================================================================================================  
ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS   Calculate mean and variance in batch normalization(BN) in single pass ( `RFC <https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm>`__ )   
=========================================  ===================================================================================================================================================================

.. warning:: 

   * Enabling experimental features does not guarantee that the library will utilize them
   
   * Enabling experimental features might change accuracy of oneDNN primitives

