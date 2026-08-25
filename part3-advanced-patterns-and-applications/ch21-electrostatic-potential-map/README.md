# Chapter 21: Electrostatic potential map

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 21 (pp. 513-528).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_dcs_scatter.cu` | §21.2, Fig. 21.4-21.5 | Direct Coulomb Summation (DCS), atom-outer scatter kernel: one thread per atom, atomic writes to every grid point |
| `02_dcs_gather.cu` | §21.2, Fig. 21.3, Fig. 21.6 | DCS gather kernel: one thread per grid point, no atomics, atoms loop innermost (unoptimized loop order) |
| `03_dcs_coarsened.cu` | §21.3, Fig. 21.7-21.8 | Gather kernel with thread coarsening: each thread computes 4 adjacent grid points, reusing per-atom constant-memory reads and the y/z distance components across them |
| `04_dcs_coalesced.cu` | §21.4, Fig. 21.9-21.10 | Same coarsening factor, but grid points are reassigned so adjacent threads write adjacent addresses -- fully coalesced energygrid writes |
| `05_dcs_cutoff_binning.cu` | §21.5, Fig. 21.11-21.14 | Cutoff summation via spatial binning: atoms sorted into bins, a fixed per-block neighborhood-bin list, shared-memory-staged distance checks |

All five files process a single fixed-`z` 2D slice of a 3D energy grid
(`GRID_X x GRID_Y = 96x96`, `GRIDSPACING = 0.5`), which is the unit of work
the chapter's own C code and kernels operate on (§21.2: "the function is
called repeatedly for all the slices of the modeled space"; a real
application would loop this per-slice kernel over the z dimension, which
is orthogonal to the optimizations this chapter demonstrates). Each file
generates `NUM_ATOMS=4000` synthetic atoms (random `x,y` inside the grid's
footprint, `charge` in `[-2,2]`) with `z` kept in `[3,9]` so no
atom-gridpoint distance is ever near zero (this repo convention: no
singular denominators corrupting the pass/fail comparison). Every file
computes an independent CPU reference (`double`-precision triple loop),
compares with `nearlyEqual` (default `eps=1e-3`), and prints `PASS`/`FAIL`
plus a `GpuTimer` timing line. Measured `max|diff|` is ~1e-5 across all
five files (`float` accumulation vs. the `double` CPU reference), well
inside the default epsilon.

## §21.2 Scatter vs. gather -- `01_dcs_scatter.cu`, `02_dcs_gather.cu`

Fig. 21.3 is the book's unoptimized DCS C code: `for(y) for(x) for(atom)`.
Fig. 21.4 loop-interchanges it to `for(atom) for(y) for(x)`, which is valid
because all iterations are independent, and lets the z- and y-distance
components be hoisted out of the two inner loops (computed once per atom,
and once per atom-row, respectively, instead of once per atom-gridpoint
pair).

- **`01_dcs_scatter.cu`** parallelizes Fig. 21.4 directly (Fig. 21.5): one
  thread per atom, each thread scatters its atom's contribution into every
  grid point of the slice. Because many threads (different atoms) can
  target the same grid point at the same time, the energygrid update must
  be `atomicAdd` -- exactly the cost the chapter calls out ("this scatter
  approach ... requires atomic operations ... which significantly reduces
  the speed of parallel execution").
- **`02_dcs_gather.cu`** instead parallelizes the *unoptimized* Fig. 21.3
  order (Fig. 21.6): one thread per grid point (a 2D thread grid matching
  the 2D potential-grid shape one-to-one), with the atoms loop innermost
  and no atomics, since each thread owns exactly one output element. This
  is the chapter's stated dilemma: the *faster* sequential C code (Fig.
  21.4) parallelizes into the *slower* kernel (needs atomics), while the
  *slower* sequential C code (Fig. 21.3) parallelizes into the *faster*
  kernel.

Both kernels share the host-side design described in §21.2: atoms are
staged into GPU **constant memory** in `CHUNK_SIZE=1500`-atom chunks (so
each chunk's byte size stays well inside the constant-memory budget), and
the kernel is invoked once per chunk, accumulating into the same device
energygrid buffer across chunks (`4000` atoms -> 3 chunks of `1500, 1500,
1000`).

Measured (release build, `DEBUG=0`, RTX-class GPU): the gather kernel
(`02`) is dramatically faster than the scatter kernel (`01`) at this
problem size, consistent with the chapter's claim that atomics are the
scatter kernel's dominant cost.

## §21.3 Thread coarsening -- `03_dcs_coarsened.cu`

Fig. 21.7 observes that grid points in the same row share the y-component
of their atom distance; Fig. 21.8 folds `COARSEN=4` grid points from one
row into a single thread so each atom's `x, y, z, charge` are read from
constant memory **once** and reused across all 4 outputs (`dy`, `dz`,
`dysqdzsq`, and `charge` computed/read once per atom; `atom.x` read once
and reused for `dx0..dx3`). This is the exact reuse pattern and operation
count the chapter itemizes ("eliminates three accesses to constant memory
for the y coordinate ... three for the x coordinate ... three for the
charge ... when processing an atom for four grid points").

This file deliberately keeps Fig. 21.8's **adjacent** grid-point-per-thread
assignment (`xBase, xBase+1, xBase+2, xBase+3`) -- the un-coalesced layout
that §21.4 identifies as the next problem to fix. `04_dcs_coalesced.cu` is
the "after" of that fix, so the two files reproduce the book's own
before/after narrative.

## §21.4 Memory coalescing -- `04_dcs_coalesced.cu`

§21.4 profiles the Fig. 21.8 kernel and finds its writes un-coalesced:
because each thread's 4 output grid points are adjacent to each other,
adjacent *threads*' outputs are 4 elements apart in the energygrid array.
Fig. 21.9's fix: assign `blockDim.x` consecutive grid points to the
threads in a block, then the next `blockDim.x` consecutive points to the
same threads, and so on -- i.e. thread `threadIdx.x` owns grid points at
offsets `threadIdx.x + i*blockDim.x` (`i = 0..COARSEN-1`) relative to the
block's base x-index, not `threadIdx.x*COARSEN + i`. With this assignment,
for any fixed `i`, addresses across `threadIdx.x` are contiguous, so all
writes from a warp are coalesced. The per-atom arithmetic and reuse are
otherwise identical to Fig. 21.8; only the index mapping (and the
resulting `blockDim.x*gridspacing`-apart x-coordinate offsets, per §21.4)
changes.

## §21.5 Cutoff binning -- `05_dcs_cutoff_binning.cu`

§21.5 replaces exact DCS (`O(atoms x gridpoints)`, which scales
quadratically with system volume) with **cutoff summation**: each grid
point sums only atoms within a fixed cutoff radius, trading a small amount
of accuracy for linear scaling. An atom-centric (scatter) cutoff kernel is
ruled out for the same atomics reason as §21.2, so the chapter builds a
grid-centric **cutoff binning** algorithm (credited to Rodrigues et al.
[3]):

1. Atoms are sorted once into spatial bins sized to match one thread
   block's grid-point footprint (§21.5, Fig. 21.13: "based on the block
   dimensions and the grid spacing, one can calculate the area ... covered
   by each block ... assume that each of these areas is also covered by a
   bin"). This file uses that correspondence literally: `BIN_DIM=8` grid
   points per bin per axis (`8 x 0.5 = 4 A` per bin, matching the book's own
   worked example: "if the grid spacing is 0.5 A and the blocks are 8x8x8,
   each block would cover a 4x4x4 cube"), and one CUDA block per bin
   (`blockIdx == bin coordinate`, exactly).
2. A fixed **neighborhood-bin list** (Fig. 21.12/21.14) -- the relative bin
   offsets that could contain an atom within cutoff distance of any grid
   point the block owns -- is computed once on the host and supplied to the
   kernel via constant memory (§21.5: "prepared before launching the grid
   ... supplied to the kernel, most likely as a constant memory array").
   This file uses `CUTOFF=12.0`, matching the chapter's own example ("a
   typical cut-off distance of 12 A for molecular-level force calculation").
3. For each neighborhood bin, the block collaboratively stages that bin's
   atoms through shared memory in `blockDim`-sized chunks (§21.5: "threads
   in a block collaborate in loading the atom information ... into shared
   memory. All threads then examine the atoms out of shared memory"); each
   thread independently checks every staged atom against its own cutoff
   distance before accumulating -- the chapter's noted source of control
   divergence ("each thread could make different decisions on including or
   excluding each atom").
4. Atoms live in **global memory**, not constant memory, per §21.5's
   reasoning: different blocks examine different neighborhoods, so
   constant memory's limited size can't hold what every active block needs
   simultaneously.

The CPU reference independently computes the same cutoff-restricted direct
summation (checking `distance <= CUTOFF` per atom-gridpoint pair, no
binning), so a match against it validates both the binning/neighborhood
machinery and the cutoff decision itself.

### Documented judgment calls for this file

- **Neighborhood-list construction.** §21.5 computes the per-block
  neighbor-bin list with a conservative "super circle" approximation
  (bin-center-to-bin-center distance <= `cutoff + half the bin diagonal`).
  This file instead computes the **exact** axis-aligned box-to-box minimum
  distance between the origin bin and each candidate neighbor bin, and
  keeps the neighbor iff that minimum distance is `<= CUTOFF`. This is a
  precise version of the same test the book's approximation exists to
  answer (list membership is still one fixed set of relative offsets per
  block, still built once on the host, still delivered via constant
  memory), and being exact rather than a conservative superset, it cannot
  itself introduce error -- the per-atom cutoff check inside the kernel is
  still the sole source of the cutoff decision. This substitution is
  called out here explicitly so a reviewer can check it against §21.5's
  text independently. With `BIN_DIM=8` (`4 A` bins) and `CUTOFF=12`, the
  computed neighborhood is 61 bins per block.
- **2D (not 3D) binning.** Every file in this chapter operates on one
  fixed-`z` slice, so bins here are indexed by `(x, y)` only, not `(x, y,
  z)` as the chapter's general description implies ("bins are implemented
  as multi-dimensional arrays: the x, y, and z dimensions"). Atom `z`
  coordinates are restricted to `[3, 9]`, close enough to the slice
  (`z=0`) that the intended cutoff-radius (`12`) mix of included/excluded
  atoms per grid point is preserved without needing a z-axis bin
  dimension.
- **No fixed-capacity bins / overflow list.** §21.5 goes on to describe a
  further data-size-scalability refinement: fixed-capacity bins (for
  memory-coalescing-friendly, uniformly sized bin storage), a host-side
  overflow list for atoms that don't fit their home bin, and hiding the
  host's overflow processing behind the next kernel launch. This file uses
  a variable-length CSR bin layout instead (`binStart`/`binCount` indexing
  into one atoms-sorted-by-bin array), which forgoes that specific
  coalescing optimization but cannot drop any atom, so no overflow
  handling is needed for correctness. This keeps the sample focused on the
  binning/neighborhood/cutoff technique itself, which is what the task
  brief for this file scopes to.

## Compute-sanitizer

All five binaries were run under `compute-sanitizer --tool memcheck` (0
errors on each). `01_dcs_scatter.cu`'s atomic scatter kernel was
additionally run under `compute-sanitizer --tool racecheck` (0 hazards),
since scatter-with-atomics is exactly the class of hazard racecheck is
built to catch.
