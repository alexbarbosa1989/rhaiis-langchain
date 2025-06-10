## LangChain workflow with RHAIIS model inference

### Set Up RHAIIS
O.S Version: Fedora 42
GPU: NVIDIA RTX 4060 ti 

If trying to start the RHAIIS container, get:
~~~
Error: default OCI runtime "nvidia" not found: invalid argument
~~~

- We need to install the NVIDIA container toolkit https://docs.nvidia.com/ai-enterprise/deployment/rhel-with-kvm/latest/podman.html
~~~
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
~~~
~~~
sudo dnf install -y nvidia-container-toolkit
~~~
- Then, reboot!

- Now, we need to set CDI https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html:
~~~
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
~~~
- List the available GPUS:
~~~
nvidia-ctk cdi list
~~~
~~~
INFO[0000] Found 3 CDI devices                          
nvidia.com/gpu=0
nvidia.com/gpu=GPU-d5732e3b-47eb-2b0c-3340-69c1fb70c0e8
nvidia.com/gpu=all
~~~

Now it is possible to follow the normal procedure from the [RHAIIS documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/getting_started/serving-and-inferencing-rhaiis_getting-started)
- Log In to the Red Hat registry
~~~
podman login registry.redhat.io
~~~
- Pull the RHAIIS image:
~~~
podman pull registry.redhat.io/rhaiis/vllm-<gpu_type>-rhel9:3.0.0
~~~
- Create a directory that will be used as container storage for RHAIIS instances:
~~~
mkdir -p rhaiis-cache/{flashinfer,huggingface,torch,vllm}
~~~
~~~
chmod a+rwX rhaiis-cache
~~~
- Set your [HugginFace token](https://huggingface.co/docs/hub/en/security-tokens) into the environment variable HF_TOKEN
~~~
echo "export HF_TOKEN=<your_HF_token>" > private.env
~~~
~~~
source private.env
~~~
- Start the RHAIIS instance
~~~
podman run -ti --rm --pull=newer \
--user 0 \
--shm-size=0 \
-p 127.0.0.1:8000:8000 \
--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
--env "HF_HUB_OFFLINE=0" \
-v ./rhaiis-cache:/opt/app-root/src/.cache  \
--device nvidia.com/gpu=all \
--security-opt=label=disable \
--name rhaiis \
registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.0.0 \
--model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 \
--max_model_len=4096
~~~

### Set up and run the application
- Clone the application repo:
~~~
git clone https://github.com/alexbarbosa1989/rhaiis-langchain.git
~~~
- Move to app directory:
~~~
cd rhaiis-langchain
~~~
- Create a `.evn` file setting the `DOCUMENT_PATH` variable with the actual PDF that will be processed:
~~~
echo "DOCUMENT_PATH=/home/user/rhaiis-langchain/docs/example/contract-template.pdf" >> .env
~~~
- Create a Python virtual environment:
~~~
python3.12 -m venv --upgrade-deps venv
~~~
- Activate it:
~~~
source venv/bin/activate
~~~
- Install the app requirements:
~~~
pip install -r requirements.txt
~~~
- Run the app:
~~~
python app.py
~~~
