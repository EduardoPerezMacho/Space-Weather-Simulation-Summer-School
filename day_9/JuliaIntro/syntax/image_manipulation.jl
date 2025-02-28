cd(@__DIR__)
using LinearAlgebra, Images, FileIO

# load image of Philip!
philip = load("philip.png")

# zoom in on head
head = philip[470:800, 140:410]

# two philip's
two_philip = [philip philip]

# four way philip
four_way_philip = [head                    reverse(head, dims = 2);
                   reverse(head, dims = 1) rot180(head)]

# Compression
rand_img = load("random_pic_1.jpeg")
U, Σ, V = svd(rand_img)
img_chain = Gray.(U[:,1]*Diagonal(Σ[1:1])*V[:,1]')
for N in 11:10:51
    compressed_img = Gray.(U[:,1:N]*Diagonal(Σ[1:N])*V[:,1:N]')
    img_chain = vcat(img_chain, compressed_img)
end
img_chain
save("approximation_strip.jpeg", img_chain)
