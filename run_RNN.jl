using Flux, DelimitedFiles
using LinearAlgebra, Random
using Distributions
using Flux: onehot, onehotbatch, onecold, throttle, kldivergence, chunk, batchseq
using Flux.Optimise: update!
using Zygote

Zygote.@nograd onehotbatch

labels = 0:1
batch_size = 10

train_path = "tfim1D_samples"
psi_path = "tfim1D_psi"

data = Int.(readdlm(train_path))
psi = readdlm(psi_path)[:,1]

train_data = data[1:10000, :]
test_data = data[10001:end, :]

N = size(train_data,2)

train_data = [train_data[i,:] for i = 1:size(train_data, 1)]
test_data = [test_data[i,:] for i = 1:size(test_data, 1)]

train_data = map(ch -> onehotbatch(ch, labels), train_data)
test_data = map(ch -> onehotbatch(ch, labels), test_data)

m = Chain(
    GRU(length(labels), 100),
    GRU(100, length(labels)),
    softmax
)
m = f64(m)
opt = ADAM(0.01)

function probability(v)
    if length(size(v)) == 1
        Flux.reset!(m)
        v0 = onehot(0, labels)
        prob = dot(m(v0), v[1])

        for i in 1:N-1
            prob *= dot(m(v[i]), v[i+1])
        end
        
    else
        v0 = onehot(0, labels)
        prob = zeros(size(v,1))
        
        for j in 1:size(v,1)
            Flux.reset!(m)
            v0 = onehot(0, labels)
            prob[j] = dot(m(v0), v[j,:][1])
            
            for i in 1:N-1
                prob[j] *= dot(m(v[j,:][i]), v[j,:][i+1])
            end
        end
    end
    
    return prob
end

function loss(v)
    # NLL
    log_prob = log(probability(v))
    return -sum(log_prob)
end

function generate_hilbert_space(;hot=false)                                               
    dim = [i for i in 0:2^N-1]                                                  
    space = space = parse.(Int64, split(bitstring(dim[1])[end-N+1:end],""))     
                                                                                
    for i in 2:length(dim)                                                      
        tmp = parse.(Int64, split(bitstring(dim[i])[end-N+1:end],""))           
        space = hcat(space, tmp)                                                
    end                                                                         
    
    if hot
        return map(ch -> onehot(ch, labels), transpose(space))
        
    else
        return transpose(space)                                                     
    end
    
end 

function sample_model(num_samples)
    
    y = zeros(num_samples, N)
    
    for s in 1:num_samples
        Flux.reset!(m)
        v0 = onehot(0, labels)
        ps = [m(v0)]
        d = Multinomial.(1, ps)
        x = Bool.(reshape(rand.(d,1)[1], (2,)))
        y[s,1] = onecold(x, labels)[1]
        
        for i in 1:N-1
            x = onecold(x, labels)[1]
            x = onehot(x, labels)
            ps = [m(x)]
            d = Multinomial.(1, ps)
            x = Bool.(reshape(rand.(d,1)[1], (2,)))
            y[s,i] = onecold(x, labels)[1]
        end
        
    end
    
    return y
end

function abs_magnetization(num_samples)
    s = sample_model(num_samples)
    s = (s .* 2) .- 1
    tmp = abs.(sum(s, dims=2) / N)
    tmp = sum(tmp) / num_samples
    return tmp
end

function fidelity(space, target)
    return dot(target, sqrt.(probability(space)))
end    

train_data = [view(train_data, k:k+batch_size-1, :) for k in 1:batch_size:size(train_data,1)];
test_data = [view(test_data, k:k+batch_size-1, :) for k in 1:batch_size:size(test_data,1)];

epochs = 1:100
num_batches = size(train_data,1) # needs to generalize
println(num_batches)
num_samples = 2000

ps = Flux.params(m)
space = generate_hilbert_space(hot=true) 

for ep in epochs
    for b in 1:num_batches
        batch = train_data[b]
        testbatch = test_data[b]
        
        gs = Flux.gradient(() -> loss_total(batch, batch), ps)
        #println(typeof(gs))
        #println(typeof(ps))
        update!(opt, ps, gs)
    
        Flux.reset!(m)

        #for p in Flux.params(m)
        #    @show size(p)
        #    @show typeof(gs[p])
        #    update!(opt, p, gs[p])
        #end
                
    end
    
    #println("loss: ",loss(test_data[1]))
    #println("abs(mag): ", abs_magnetization(num_samples))
    println("fidelity: ", fidelity(space, psi))
    
end
