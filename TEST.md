<vllm>
```
vLLM OpenAI-Compatible RESTful API server.
  --quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,experts_int8,None}, -q {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,experts_int8,None}
                        Method used to quantize the weights. If None, we first check the `quantization_config` attribute in the model config file. If that is None, we assume the model weights are not quantized
                        and use `dtype` to determine the data type of the weights.
  --rope-scaling ROPE_SCALING
                        RoPE scaling configuration in JSON format. For example, {"type":"dynamic","factor":2.0}
  --rope-theta ROPE_THETA
                        RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE theta improves the performance of the scaled model.
  --enforce-eager       Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility.
  --max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE
                        Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. (DEPRECATED. Use --max-seq-len-to-capture instead)
  --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode.
  --disable-custom-all-reduce
                        See ParallelConfig.
  --tokenizer-pool-size TOKENIZER_POOL_SIZE
                        Size of tokenizer pool to use for asynchronous tokenization. If 0, will use synchronous tokenization.
  --tokenizer-pool-type TOKENIZER_POOL_TYPE
                        Type of tokenizer pool to use for asynchronous tokenization. Ignored if tokenizer_pool_size is 0.
  --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
                        Extra config for tokenizer pool. This should be a JSON string that will be parsed into a dictionary. Ignored if tokenizer_pool_size is 0.
  --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
                        For each multimodal plugin, limit how many input instances to allow for each prompt. Expects a comma-separated list of items, e.g.: `image=16,video=2` allows a maximum of 16 images and 2
                        videos per prompt. Defaults to 1 for each modality.
  --enable-lora         If True, enable handling of LoRA adapters.
  --max-loras MAX_LORAS
                        Max number of LoRAs in a single batch.
  --max-lora-rank MAX_LORA_RANK
                        Max LoRA rank.
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base model vocabulary).
  --lora-dtype {auto,float16,bfloat16,float32}
                        Data type for LoRA. If auto, will default to base model dtype.
  --long-lora-scaling-factors LONG_LORA_SCALING_FACTORS
                        Specify multiple scaling factors (which can be different from base model scaling factor - see eg. Long LoRA) to allow for multiple LoRA adapters trained with those scaling factors to be
                        used at the same time. If not specified, only adapters trained with the base model scaling factor are allowed.
  --max-cpu-loras MAX_CPU_LORAS
                        Maximum number of LoRAs to store in CPU memory. Must be >= than max_num_seqs. Defaults to max_num_seqs.
  --fully-sharded-loras
                        By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers. At high sequence length, max rank or tensor parallel
                        size, this is likely faster.
  --enable-prompt-adapter
                        If True, enable handling of PromptAdapters.
  --max-prompt-adapters MAX_PROMPT_ADAPTERS
                        Max number of PromptAdapters in a batch.
  --max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN
                        Max number of PromptAdapters tokens
  --device {auto,cuda,neuron,cpu,openvino,tpu,xpu}
                        Device type for vLLM execution.
  --num-scheduler-steps NUM_SCHEDULER_STEPS
                        Maximum number of forward steps per scheduler call.
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
                        Apply a delay (of delay factor multiplied by previousprompt latency) before scheduling next prompt.
  --enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]
                        If set, the prefill requests can be chunked based on the max_num_batched_tokens.
  --speculative-model SPECULATIVE_MODEL
                        The name of the draft model to be used in speculative decoding.
  --speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,squeezellm,compressed-tensors,bitsandbytes,qqq,experts_int8,None}
                        Method used to quantize the weights of speculative model.If None, we first check the `quantization_config` attribute in the model config file. If that is None, we assume the model weights
                        are not quantized and use `dtype` to determine the data type of the weights.
  --num-speculative-tokens NUM_SPECULATIVE_TOKENS
                        The number of speculative tokens to sample from the draft model in speculative decoding.
  --speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE, -spec-draft-tp SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas for the draft model in speculative decoding.
  --speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN
                        The maximum sequence length supported by the draft model. Sequences over this length will skip speculation.
  --speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE
                        Disable speculative decoding for new incoming requests if the number of enqueue requests is larger than this value.
  --ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX
                        Max size of window for ngram prompt lookup in speculative decoding.
  --ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN
                        Min size of window for ngram prompt lookup in speculative decoding.
  --spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}
                        Specify the acceptance method to use during draft token verification in speculative decoding. Two types of acceptance routines are supported: 1) RejectionSampler which does not allow
                        changing the acceptance rate of draft tokens, 2) TypicalAcceptanceSampler which is configurable, allowing for a higher acceptance rate at the cost of lower quality, and vice versa.
  --typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD
                        Set the lower bound threshold for the posterior probability of a token to be accepted. This threshold is used by the TypicalAcceptanceSampler to make sampling decisions during speculative
                        decoding. Defaults to 0.09
  --typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA
                        A scaling factor for the entropy-based threshold for token acceptance in the TypicalAcceptanceSampler. Typically defaults to sqrt of --typical-acceptance-sampler-posterior-threshold i.e.
                        0.3
  --disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]
                        If set to True, token log probabilities are not returned during speculative decoding. If set to False, log probabilities are returned according to the settings in SamplingParams. If not
                        specified, it defaults to True. Disabling log probabilities during speculative decoding reduces latency by skipping logprob calculation in proposal sampling, target sampling, and after
                        accepted tokens are determined.
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. This should be a JSON string that will be parsed into a dictionary.
  --ignore-patterns IGNORE_PATTERNS
                        The pattern(s) to ignore when loading the model.Default to 'original/**/*' to avoid repeated loading of llama's checkpoints.
  --preemption-mode PREEMPTION_MODE
                        If 'recompute', the engine performs preemption by recomputing; If 'swap', the engine performs preemption by block swapping.
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API. If multiple names are provided, the server will respond to any of the provided names. The model name in the model field of a response will be the first
                        name in this list. If not specified, the model name will be the same as the `--model` argument. Noted that this name(s)will also be used in `model_name` tag content of prometheus metrics,
                        if multiple names provided, metricstag will take the first one.
  --qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH
                        Name or path of the QLoRA adapter.
  --otlp-traces-endpoint OTLP_TRACES_ENDPOINT
                        Target URL to which OpenTelemetry traces will be sent.
  --collect-detailed-traces COLLECT_DETAILED_TRACES
                        Valid choices are model,worker,all. It makes sense to set this only if --otlp-traces-endpoint is set. If set, it will collect detailed traces for the specified modules. This involves use
                        of possibly costly and or blocking operations and hence might have a performance impact.
  --engine-use-ray      Use Ray to start the LLM engine in a separate process as the server process.(DEPRECATED. This argument is deprecated and will be removed in a future update. Set
                        `VLLM_ALLOW_ENGINE_USE_RAY=1` to force use it. See https://github.com/vllm-project/vllm/issues/7045.)
  --disable-log-requests
                        Disable logging requests.
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers being printed in log. Default: Unlimited
```
</vllm>

<triton inference server>
```
tritonserver [options]
  --id <string>: Identifier for this server.
  
  --model-repository <string>
        Path to model repository directory. It may be specified
        multiple times to add multiple model repositories. Note that if a model
        is not unique across all model repositories at any time, the model
        will not be available.

  --exit-on-error <boolean>
        Exit the inference server if an error occurs during
        initialization.
        
  --disable-auto-complete-config
        If set, disables the triton and backends from auto
        completing model configuration files. Model configuration files must be
        provided and all required configuration settings must be specified.
  --strict-readiness <boolean>
        If true /v2/health/ready endpoint indicates ready if the
        server is responsive and all models are available. If false
        /v2/health/ready endpoint indicates ready if server is responsive even if
        some/all models are unavailable.
  --allow-http <boolean>
        Allow the server to listen for HTTP requests.
  --http-port <integer>
        The port for the server to listen on for HTTP requests.
  --reuse-http-port <boolean>
        Allow multiple servers to listen on the same HTTP port when
        every server has this option set. If you plan to use this option as
        a way to load balance between different Triton servers, the same
        model repository or set of models must be used for every server.
  --http-address <string>
        The address for the http server to binds to.
  --http-thread-count <integer>
        Number of threads handling HTTP requests.
  --allow-grpc <boolean>
        Allow the server to listen for GRPC requests.
  --grpc-port <integer>
        The port for the server to listen on for GRPC requests.
  --reuse-grpc-port <boolean>
        Allow multiple servers to listen on the same GRPC port when
        every server has this option set. If you plan to use this option as
        a way to load balance between different Triton servers, the same
        model repository or set of models must be used for every server.
  --grpc-address <string>
        The address for the grpc server to binds to.
  --grpc-infer-allocation-pool-size <integer>
        The maximum number of inference request/response objects
        that remain allocated for reuse. As long as the number of in-flight
        requests doesn't exceed this value there will be no
        allocation/deallocation of request/response objects.
  --grpc-use-ssl <boolean>
        Use SSL authentication for GRPC requests. Default is false.
  --grpc-use-ssl-mutual <boolean>
        Use mututal SSL authentication for GRPC requests. Default is
        false.
  --grpc-server-cert <string>
        File holding PEM-encoded server certificate. Ignored unless
        --grpc-use-ssl is true.
  --grpc-server-key <string>
        File holding PEM-encoded server key. Ignored unless
        --grpc-use-ssl is true.
  --grpc-root-cert <string>
        File holding PEM-encoded root certificate. Ignore unless
        --grpc-use-ssl is false.
  --grpc-infer-response-compression-level <string>
        The compression level to be used while returning the infer
        response to the peer. Allowed values are none, low, medium and high.
        By default, compression level is selected as none.
  --grpc-keepalive-time <integer>
        The period (in milliseconds) after which a keepalive ping is
        sent on the transport. Default is 7200000 (2 hours).
  --grpc-keepalive-timeout <integer>
        The period (in milliseconds) the sender of the keepalive
        ping waits for an acknowledgement. If it does not receive an
        acknowledgment within this time, it will close the connection. Default is
        20000 (20 seconds).
  --grpc-keepalive-permit-without-calls <boolean>
        Allows keepalive pings to be sent even if there are no calls
        in flight (0 : false; 1 : true). Default is 0 (false).
  --grpc-http2-max-pings-without-data <integer>
        The maximum number of pings that can be sent when there is
        no data/header frame to be sent. gRPC Core will not continue sending
        pings if we run over the limit. Setting it to 0 allows sending pings
        without such a restriction. Default is 2.
  --grpc-http2-min-recv-ping-interval-without-data <integer>
        If there are no data/header frames being sent on the
        transport, this channel argument on the server side controls the minimum
        time (in milliseconds) that gRPC Core would expect between receiving
        successive pings. If the time between successive pings is less than
        this time, then the ping will be considered a bad ping from the peer.
        Such a ping counts as a ‘ping strike’. Default is 300000 (5
        minutes).
  --grpc-http2-max-ping-strikes <integer>
        Maximum number of bad pings that the server will tolerate
        before sending an HTTP2 GOAWAY frame and closing the transport.
        Setting it to 0 allows the server to accept any number of bad pings.
        Default is 2.
  --allow-sagemaker <boolean>
        Allow the server to listen for Sagemaker requests. Default
        is false.
  --sagemaker-port <integer>
        The port for the server to listen on for Sagemaker requests.
        Default is 8080.
  --sagemaker-safe-port-range <<integer>-<integer>>
        Set the allowed port range for endpoints other than the
        SageMaker endpoints.
  --sagemaker-thread-count <integer>
        Number of threads handling Sagemaker requests. Default is 8.
  --allow-vertex-ai <boolean>
        Allow the server to listen for Vertex AI requests. Default
        is true if AIP_MODE=PREDICTION, false otherwise.
  --vertex-ai-port <integer>
        The port for the server to listen on for Vertex AI requests.
        Default is AIP_HTTP_PORT if set, 8080 otherwise.
  --vertex-ai-thread-count <integer>
        Number of threads handling Vertex AI requests. Default is 8.
  --vertex-ai-default-model <string>
        The name of the model to use for single-model inference
        requests.
  --allow-metrics <boolean>
        Allow the server to provide prometheus metrics.
  --allow-gpu-metrics <boolean>
        Allow the server to provide GPU metrics. Ignored unless
        --allow-metrics is true.
  --allow-cpu-metrics <boolean>
        Allow the server to provide CPU metrics. Ignored unless
        --allow-metrics is true.
  --metrics-port <integer>
        The port reporting prometheus metrics.
  --metrics-interval-ms <float>
        Metrics will be collected once every <metrics-interval-ms>
        milliseconds. Default is 2000 milliseconds.
  --trace-file <string>
        Set the file where trace output will be saved. If
        --trace-log-frequency is also specified, this argument value will be the
        prefix of the files to save the trace output. See --trace-log-frequency
        for detail.
  --trace-level <string>
        Specify a trace level. OFF to disable tracing, TIMESTAMPS to
        trace timestamps, TENSORS to trace tensors. It may be specified
        multiple times to trace multiple informations. Default is OFF.
  --trace-rate <integer>
        Set the trace sampling rate. Default is 1000.
  --trace-count <integer>
        Set the number of traces to be sampled. If the value is -1,
        the number of traces to be sampled will not be limited. Default is
        -1.
  --trace-log-frequency <integer>
        Set the trace log frequency. If the value is 0, Triton will
        only log the trace output to <trace-file> when shutting down.
        Otherwise, Triton will log the trace output to <trace-file>.<idx> when it
        collects the specified number of traces. For example, if the log
        frequency is 100, when Triton collects the 100-th trace, it logs the
        traces to file <trace-file>.0, and when it collects the 200-th trace,
        it logs the 101-th to the 200-th traces to file <trace-file>.1.
        Default is 0.
  --model-control-mode <string>
        Specify the mode for model management. Options are "none",
        "poll" and "explicit". The default is "none". For "none", the server
        will load all models in the model repository(s) at startup and will
        not make any changes to the load models after that. For "poll", the
        server will poll the model repository(s) to detect changes and will
        load/unload models based on those changes. The poll rate is
        controlled by 'repository-poll-secs'. For "explicit", model load and unload
        is initiated by using the model control APIs, and only models
        specified with --load-model will be loaded at startup.
  --repository-poll-secs <integer>
        Interval in seconds between each poll of the model
        repository to check for changes. Valid only when --model-control-mode=poll is
        specified.
  --load-model <string>
        Name of the model to be loaded on server startup. It may be
        specified multiple times to add multiple models. To load ALL models
        at startup, specify '*' as the model name with --load-model=* as the
        ONLY --load-model argument, this does not imply any pattern
        matching. Specifying --load-model=* in conjunction with another
        --load-model argument will result in error. Note that this option will only
        take effect if --model-control-mode=explicit is true.
  --rate-limit <string>
        Specify the mode for rate limiting. Options are
        "execution_count" and "off". The default is "off". For "execution_count", the
        server will determine the instance using configured priority and the
        number of time the instance has been used to run inference. The
        inference will finally be executed once the required resources are
        available. For "off", the server will ignore any rate limiter config and
        run inference as soon as an instance is ready.
  --rate-limit-resource <<string>:<integer>:<integer>>
        The number of resources available to the server. The format
        of this flag is
        --rate-limit-resource=<resource_name>:<count>:<device>. The <device> is optional and if not listed will be applied to
        every device. If the resource is specified as "GLOBAL" in the model
        configuration the resource is considered shared among all the devices
        in the system. The <device> property is ignored for such resources.
        This flag can be specified multiple times to specify each resources
        and their availability. By default, the max across all instances
        that list the resource is selected as its availability. The values for
        this flag is case-insensitive.
  --pinned-memory-pool-byte-size <integer>
        The total byte size that can be allocated as pinned system
        memory. If GPU support is enabled, the server will allocate pinned
        system memory to accelerate data transfer between host and devices
        until it exceeds the specified byte size. If 'numa-node' is configured
        via --host-policy, the pinned system memory of the pool size will be
        allocated on each numa node. This option will not affect the
        allocation conducted by the backend frameworks. Default is 256 MB.
  --cuda-memory-pool-byte-size <<integer>:<integer>>
        The total byte size that can be allocated as CUDA memory for
        the GPU device. If GPU support is enabled, the server will allocate
        CUDA memory to minimize data transfer between host and devices
        until it exceeds the specified byte size. This option will not affect
        the allocation conducted by the backend frameworks. The argument
        should be 2 integers separated by colons in the format <GPU device
        ID>:<pool byte size>. This option can be used multiple times, but only
        once per GPU device. Subsequent uses will overwrite previous uses for
        the same GPU device. Default is 64 MB.
  --response-cache-byte-size <integer>
        The size in bytes to allocate for a request/response cache.
        When non-zero, Triton allocates the requested size in CPU memory and
        shares the cache across all inference requests and across all
        models. For a given model to use request caching, the model must enable
        request caching in the model configuration. By default, no model uses
        request caching even if the request cache is enabled with the
        --response-cache-byte-size flag. Default is 0.
  --min-supported-compute-capability <float>
        The minimum supported CUDA compute capability. GPUs that
        don't support this compute capability will not be used by the server.
  --exit-timeout-secs <integer>
        Timeout (in seconds) when exiting to wait for in-flight
        inferences to finish. After the timeout expires the server exits even if
        inferences are still in flight.
  --backend-directory <string>
        The global directory searched for backend shared libraries.
        Default is '/opt/tritonserver/backends'.
  --repoagent-directory <string>
        The global directory searched for repository agent shared
        libraries. Default is '/opt/tritonserver/repoagents'.
  --buffer-manager-thread-count <integer>
        The number of threads used to accelerate copies and other
        operations required to manage input and output tensor contents.
        Default is 0.
  --model-load-thread-count <integer>
        The number of threads used to concurrently load models in
        model repositories. Default is 2*<num_cpu_cores>.
  --backend-config <<string>,<string>=<string>>
        Specify a backend-specific configuration setting. The format
        of this flag is --backend-config=<backend_name>,<setting>=<value>.
        Where <backend_name> is the name of the backend, such as 'tensorrt'.
  --host-policy <<string>,<string>=<string>>
        Specify a host policy setting associated with a policy name.
        The format of this flag is
        --host-policy=<policy_name>,<setting>=<value>. Currently supported settings are 'numa-node', 'cpu-cores'.
        Note that 'numa-node' setting will affect pinned memory pool behavior,
        see --pinned-memory-pool for more detail.
  --model-load-gpu-limit <<device_id>:<fraction>>
        Specify the limit on GPU memory usage as a fraction. If
        model loading on the device is requested and the current memory usage
        exceeds the limit, the load will be rejected. If not specified, the
        limit will not be set.
</triton inference server>
<tgi>
Text Generation Launcher

Usage: text-generation-launcher [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers

          [env: MODEL_ID=]
          [default: bigscience/bloom-560m]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: REVISION=]

      --validation-workers <VALIDATION_WORKERS>
          The number of tokenizer workers used for payload validation and truncation inside the router

          [env: VALIDATION_WORKERS=]
          [default: 2]

      --sharded <SHARDED>
          Whether to shard the model across multiple GPUs By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard`

          [env: SHARDED=]
          [possible values: true, false]

      --num-shard <NUM_SHARD>
          The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance

          [env: NUM_SHARD=]

      --quantize <QUANTIZE>
          Whether you want the model to be quantized

          [env: QUANTIZE=]

          Possible values:
          - awq:              4 bit quantization. Requires a specific AWQ quantized model: <https://hf.co/models?search=awq>. Should replace GPTQ models wherever possible because of the better latency
          - eetq:             8 bit quantization, doesn't require specific model. Should be a drop-in replacement to bitsandbytes with much better performance. Kernels are from <https://github.com/NetEase-FuXi/EETQ.git>
          - exl2:             Variable bit quantization. Requires a specific EXL2 quantized model: <https://hf.co/models?search=exl2>. Requires exllama2 kernels and does not support tensor parallelism (num_shard > 1)
          - gptq:             4 bit quantization. Requires a specific GTPQ quantized model: <https://hf.co/models?search=gptq>. text-generation-inference will use exllama (faster) kernels wherever possible, and use triton kernel (wider support) when it's not. AWQ has faster kernels
          - marlin:           4 bit quantization. Requires a specific Marlin quantized model: <https://hf.co/models?search=marlin>
          - bitsandbytes:     Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half, but it is known that the model will be much slower to run than the native f16
          - bitsandbytes-nf4: Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x, but it is known that the model will be much slower to run than the native f16
          - bitsandbytes-fp4: Bitsandbytes 4bit. nf4 should be preferred in most cases but maybe this one has better perplexity performance for you model
          - fp8:              [FP8](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) (e4m3) works on H100 and above This dtype has native ops should be the fastest if available. This is currently not the fastest because of local unpacking + padding to satisfy matrix multiplication limitations

      --speculate <SPECULATE>
          The number of input_ids to speculate on If using a medusa model, the heads will be picked up automatically Other wise, it will use n-gram speculation which is relatively free in terms of compute, but the speedup heavily depends on the task

          [env: SPECULATE=]

      --dtype <DTYPE>
          The dtype to be forced upon the model. This option cannot be used with `--quantize`

          [env: DTYPE=]
          [possible values: float16, bfloat16]

      --trust-remote-code
          Whether you want to execute hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision

          [env: TRUST_REMOTE_CODE=]

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 128]

      --max-best-of <MAX_BEST_OF>
          This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence

          [env: MAX_BEST_OF=]
          [default: 2]

      --max-stop-sequences <MAX_STOP_SEQUENCES>
          This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt

          [env: MAX_STOP_SEQUENCES=]
          [default: 4]

      --max-top-n-tokens <MAX_TOP_N_TOKENS>
          This is the maximum allowed value for clients to set `top_n_tokens`. `top_n_tokens` is used to return information about the the `n` most likely tokens at each generation step, instead of just the sampled token. This information can be used for downstream tasks like for classification or ranking

          [env: MAX_TOP_N_TOKENS=]
          [default: 5]

      --max-input-tokens <MAX_INPUT_TOKENS>
          This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle. Default to min(max_position_embeddings - 1, 4095)

          [env: MAX_INPUT_TOKENS=]

      --max-input-length <MAX_INPUT_LENGTH>
          Legacy version of [`Args::max_input_tokens`]

          [env: MAX_INPUT_LENGTH=]

      --max-total-tokens <MAX_TOTAL_TOKENS>
          This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be. Default to min(max_position_embeddings, 4096)

          [env: MAX_TOTAL_TOKENS=]

      --waiting-served-ratio <WAITING_SERVED_RATIO>
          This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run.

          This setting is only applied if there is room in the batch as defined by `max_batch_total_tokens`.

          [env: WAITING_SERVED_RATIO=]
          [default: 0.3]

      --max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>
          Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent. Default to `max_input_tokens + 50` to give a bit of room

          [env: MAX_BATCH_PREFILL_TOKENS=]

      --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch. When using padding (not recommended) this would be equivalent of `batch_size` * `max_total_tokens`.

          However in the non-padded (flash attention) version this can be much finer.

          For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible amount that fits the remaining memory (after the model is loaded). Since the actual memory overhead depends on other parameters like if you're using quantization, flash attention or the model implementation, text-generation-inference cannot infer this number automatically.

          [env: MAX_BATCH_TOTAL_TOKENS=]

      --max-waiting-tokens <MAX_WAITING_TOKENS>
          This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch.

          With a value too small, queries will always "steal" the compute to run `prefill` and running queries will be delayed by a lot.

          With a value too big, waiting queries could wait for a very long time before being allowed a slot in the running batch. If your server is busy that means that requests that could run in ~2s on an empty server could end up running in ~20s because the query had to wait for 18s.

          This number is expressed in number of tokens to make it a bit more "model" agnostic, but what should really matter is the overall latency for end users.

          [env: MAX_WAITING_TOKENS=]
          [default: 20]

      --max-batch-size <MAX_BATCH_SIZE>
          Enforce a maximum number of requests per batch Specific flag for hardware targets that do not support unpadded inference

          [env: MAX_BATCH_SIZE=]

      --cuda-graphs <CUDA_GRAPHS>
          Specify the batch sizes to compute cuda graphs for. Use "0" to disable. Default = "1,2,4,8,16,32"

          [env: CUDA_GRAPHS=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=ddce2ef2ca87]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: PORT=80]
          [default: 3000]

      --shard-uds-path <SHARD_UDS_PATH>
          The name of the socket for gRPC communication between the webserver and the shards

          [env: SHARD_UDS_PATH=]
          [default: /tmp/text-generation-server]

      --master-addr <MASTER_ADDR>
          The address the master shard will listen on. (setting used by torch distributed)

          [env: MASTER_ADDR=]
          [default: localhost]

      --master-port <MASTER_PORT>
          The address the master port will listen on. (setting used by torch distributed)

          [env: MASTER_PORT=]
          [default: 29500]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: HUGGINGFACE_HUB_CACHE=/data]

      --weights-cache-override <WEIGHTS_CACHE_OVERRIDE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: WEIGHTS_CACHE_OVERRIDE=]

      --disable-custom-kernels
          For some models (like bloom), text-generation-inference implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues

          [env: DISABLE_CUSTOM_KERNELS=]

      --cuda-memory-fraction <CUDA_MEMORY_FRACTION>
          Limit the CUDA available memory. The allowed value equals the total visible memory multiplied by cuda-memory-fraction

          [env: CUDA_MEMORY_FRACTION=]
          [default: 1.0]

      --rope-scaling <ROPE_SCALING>
          Rope scaling will only be used for RoPE models and allow rescaling the position rotary to accomodate for larger prompts.

          Goes together with `rope_factor`.

          `--rope-factor 2.0` gives linear scaling with a factor of 2.0 `--rope-scaling dynamic` gives dynamic scaling with a factor of 1.0 `--rope-scaling linear` gives linear scaling with a factor of 1.0 (Nothing will be changed basically)

          `--rope-scaling linear --rope-factor` fully describes the scaling you want

          [env: ROPE_SCALING=]
          [possible values: linear, dynamic]

      --rope-factor <ROPE_FACTOR>
          Rope scaling will only be used for RoPE models See `rope_scaling`

          [env: ROPE_FACTOR=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          [env: OTLP_SERVICE_NAME=]
          [default: text-generation-inference.router]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: CORS_ALLOW_ORIGIN=]

      --watermark-gamma <WATERMARK_GAMMA>
          [env: WATERMARK_GAMMA=]

      --watermark-delta <WATERMARK_DELTA>
          [env: WATERMARK_DELTA=]

      --ngrok
          Enable ngrok tunneling

          [env: NGROK=]

      --ngrok-authtoken <NGROK_AUTHTOKEN>
          ngrok authentication token

          [env: NGROK_AUTHTOKEN=]

      --ngrok-edge <NGROK_EDGE>
          ngrok edge

          [env: NGROK_EDGE=]

      --tokenizer-config-path <TOKENIZER_CONFIG_PATH>
          The path to the tokenizer config file. This path is used to load the tokenizer configuration which may include a `chat_template`. If not provided, the default config will be used from the model hub

          [env: TOKENIZER_CONFIG_PATH=]

      --disable-grammar-support
          Disable outlines grammar constrained generation. This is a feature that allows you to generate text that follows a specific grammar

          [env: DISABLE_GRAMMAR_SUPPORT=]

  -e, --env
          Display a lot of information about your runtime environment

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: MAX_CLIENT_BATCH_SIZE=]
          [default: 4]

      --lora-adapters <LORA_ADAPTERS>
          Lora Adapters a list of adapter ids i.e. `repo/adapter1,repo/adapter2` to load during startup that will be available to callers via the `adapter_id` field in a request

          [env: LORA_ADAPTERS=]

      --disable-usage-stats
          Disable sending of all usage statistics

          [env: DISABLE_USAGE_STATS=]

      --disable-crash-reports
          Disable sending of crash reports, but allow anonymous usage statistics

          [env: DISABLE_CRASH_REPORTS=]
</tgi>
<tei>
Text Embedding Webserver

Usage: text-embeddings-router-90 [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `BAAI/bge-large-en-v1.5`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers

          [env: MODEL_ID=]
          [default: BAAI/bge-large-en-v1.5]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Default to the number of CPU cores on the machine

          [env: TOKENIZATION_WORKERS=]

      --dtype <DTYPE>
          The dtype to be forced upon the model

          [env: DTYPE=]
          [possible values: float16, float32]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: POOLING=]

          Possible values:
          - cls:        Select the CLS token as embedding
          - mean:       Apply Mean pooling to the model embeddings
          - splade:     Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings. This option is only available if the loaded model is a `ForMaskedLM` Transformer model
          - last-token: Select the last token as embedding

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.

          [env: MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: MAX_BATCH_REQUESTS=]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --auto-truncate
          Automatically truncate inputs that are longer than the maximum supported size

          Unused for gRPC servers

          [env: AUTO_TRUNCATE=]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: DEFAULT_PROMPT=]

      --hf-api-token <HF_API_TOKEN>
          Your HuggingFace hub token

          [env: HF_API_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=ier2024090009-dp-7c665f4cc7-kxv67]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: PORT=80]
          [default: 3000]

      --uds-path <UDS_PATH>
          The name of the unix socket some text-embeddings-inference backends will use as they communicate internally with gRPC

          [env: UDS_PATH=]
          [default: /tmp/text-embeddings-inference-server]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: HUGGINGFACE_HUB_CACHE=/data]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.

          [env: API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `text-embeddings-inference.server`

          [env: OTLP_SERVICE_NAME=]
          [default: text-embeddings-inference.server]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          Unused for gRPC servers

          [env: CORS_ALLOW_ORIGIN=]
<tei>
