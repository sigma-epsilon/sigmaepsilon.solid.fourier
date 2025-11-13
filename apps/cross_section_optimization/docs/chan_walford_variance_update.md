# 🧮 Chan–Welford Parallel Variance Update

## Overview

When processing large datasets in **batches** (e.g. from data loaders or streaming sources), we often want to compute the **mean** and **variance** without storing all samples in memory.  
The **Chan–Welford algorithm** provides a numerically stable way to do this **incrementally** or **in parallel**.

## The Goal

We want to compute:
\[
\text{mean} = \frac{1}{N} \sum_{i=1}^{N} x_i, \qquad 
\text{variance} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \text{mean})^2
\]

but we want to **update** these values as new batches arrive, without reprocessing all previous data.

## Welford’s Online Algorithm (Single-Stream)

For a single running stream of data points \( x_1, x_2, \dots, x_N \):

1. Initialize  
   \[
   n = 0, \quad \text{mean} = 0, \quad M_2 = 0
   \]
2. For each new value \( x \):
   \[
   n \leftarrow n + 1
   \]
   \[
   \delta = x - \text{mean}
   \]
   \[
   \text{mean} \leftarrow \text{mean} + \frac{\delta}{n}
   \]
   \[
   M_2 \leftarrow M_2 + \delta \cdot (x - \text{mean})
   \]
3. After processing all samples:
   \[
   \text{variance} = \frac{M_2}{n}
   \]

This algorithm maintains stability by never directly summing large squared differences.

## Chan’s Parallel Update (Combining Two Batches)

Suppose we have two disjoint data batches **A** and **B**, with statistics:

- For **A**:  
  \( n_A, \text{mean}_A, M_{2A} \)
- For **B**:  
  \( n_B, \text{mean}_B, M_{2B} \)

We can merge them into a single set **AB** without re-reading the raw data.

### Combined mean

\[
\text{mean}_{AB} = \frac{n_A \, \text{mean}_A + n_B \, \text{mean}_B}{n_A + n_B}
\]

### Mean difference

\[
\delta = \text{mean}_B - \text{mean}_A
\]

### Combined M₂ term (sum of squared deviations)

\[
M_{2,AB} = M_{2A} + M_{2B} + \delta^2 \cdot \frac{n_A \, n_B}{n_A + n_B}
\]

### Combined count

\[
n_{AB} = n_A + n_B
\]

Finally:
\[
\text{variance}_{AB} = \frac{M_{2,AB}}{n_{AB}}
\]

## Why It’s Useful

- ✅ **Parallelizable** — each worker (or batch) can compute its local `mean` and `M2`, then merge results efficiently.  
- ✅ **Streaming-friendly** — supports online updates one sample or batch at a time.  
- ✅ **Numerically stable** — avoids catastrophic cancellation from subtracting large numbers.  
- ✅ **Memory efficient** — only requires storing `n`, `mean`, and `M2`.

## References

- _B. P. Welford (1962)*_ “Note on a Method for Calculating Corrected Sums of Squares and Products.”  
- _T. F. Chan et al. (1979)_, “Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.”  
- _Donald E. Knuth (1998)_, _The Art of Computer Programming, Vol. 2_, §4.2.2.
