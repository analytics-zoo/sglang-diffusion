export http_proxy=http://proxy.iil.intel.com:911
export https_proxy=http://proxy.iil.intel.com:911
export no_proxy=localhost,127.0.0.1,10.240.203.53

pip install -e "python[diffusion]" --no-deps 
