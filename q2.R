library(datasets)
library(cluster)
library(Rtsne)
library(mclust)

data("iris")

X_raw <- iris[, 1:4]
y <- iris$Species

X <- scale(X_raw)
set.seed(123)

## Part 1
k_vals <- 2:6
silhouette_scores <- numeric(length(k_vals))

for (j in seq_along(k_vals)) {
  k <- k_vals[j]
  km <- kmeans(X, centers = k, nstart = 50)
  sil <- silhouette(km$cluster, dist(X))
  silhouette_scores[j] <- mean(sil[, "sil_width"])
}

silhouette_results <- data.frame(
  k = k_vals,
  average_silhouette = silhouette_scores
)

k_max <- silhouette_results$k[
  which.max(silhouette_results$average_silhouette)
]

set.seed(123)
km_max <- kmeans(X, centers = k_max, nstart = 100)
km3 <- kmeans(X, centers = 3, nstart = 50)

## Part 2
cat("\nContingency table: silhouette-selected clusters vs species\n")
print(table(Cluster = km_max$cluster, Species = y))

cat("\nContingency table: k=3 clusters vs species\n")
print(table(Cluster = km3$cluster, Species = y))

ari <- adjustedRandIndex(km_max$cluster, y)
cat("\nAdjusted Rand index, silhouette-selected solution:", ari, "\n")

ari3 <- adjustedRandIndex(km3$cluster, y)
cat("\nAdjusted Rand index, k=3 solution:", ari3, "\n")

## Part 3
pca <- prcomp(X)
percent_variance <- 100 * pca$sdev^2 / sum(pca$sdev^2)
pca_scores <- as.data.frame(pca$x)

cluster_colours <- c('red', 'blue','green')

pdf('PCA.pdf')
plot(
  pca_scores$PC1,
  pca_scores$PC2,
  col = cluster_colours[km3$cluster],
  xlab = paste0("PC1 (", round(percent_variance[1], 1), "%)"),
  ylab = paste0("PC2 (", round(percent_variance[2], 1), "%)"),
  main = "PCA with K-means cluster labels (k = 3)"
)
dev.off()

## Part 4
derived_raw <- data.frame(
  sepal_area = iris$Sepal.Length * iris$Sepal.Width,
  petal_area = iris$Petal.Length * iris$Petal.Width,
  sepal_ratio = iris$Sepal.Length / iris$Sepal.Width,
  petal_ratio = iris$Petal.Length / iris$Petal.Width,
  petal_to_sepal_length = iris$Petal.Length / iris$Sepal.Length,
  petal_to_sepal_width = iris$Petal.Width / iris$Sepal.Width
)

derived <- scale(derived_raw)

silhouette_derived <- numeric(length(k_vals))
for (j in seq_along(k_vals)) {
  k <- k_vals[j]
  km <- kmeans(derived, centers = k, nstart = 50)
  sil <- silhouette(km$cluster, dist(derived))
  silhouette_derived[j] <- mean(sil[, "sil_width"])
}

derived_results <- data.frame(
  k = k_vals,
  average_silhouette = silhouette_derived
)

k_derived <- derived_results$k[which.max(derived_results$average_silhouette)]
set.seed(123)
km_derived <- kmeans(derived, centers = k_derived, nstart = 100)
km3_derived <- kmeans(derived, centers = 3, nstart = 100)

cat("\nContingency table: derived-variable clusters vs species\n")
print(table(Cluster = km3_derived$cluster, Species = y))

cat("Adjusted Rand index for derived-variable clustering:",
    adjustedRandIndex(km3_derived$cluster, y), "\n")

## Part 5
perplexities <- c(5, 15, 30, 45)
tsne_results <- vector("list", length(perplexities))

for (j in seq_along(perplexities)) {
  set.seed(100 + perplexities[j])
  tsne_results[[j]] <- Rtsne(
    X,
    dims = 2,
    perplexity = perplexities[j],
    theta = 0.5,
    max_iter = 1000,
    check_duplicates = FALSE,
    pca = FALSE,
    verbose = FALSE
  )
}

par(mfrow = c(2, 2))
pdf('tsne.pdf')

for (j in seq_along(perplexities)) {
  tsne_xy <- tsne_results[[j]]$Y
  plot(
    tsne_xy,
    col = c('red', 'blue', 'green')[y],
    xlab = "t-SNE dimension 1",
    ylab = "t-SNE dimension 2",
    main = paste("t-SNE, perplexity =", perplexities[j])
  )
}
dev.off()
par(mfrow = c(1, 1))
