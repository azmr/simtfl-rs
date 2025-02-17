//! # SimTFL-RS
//!
//! A work-in-progress simulator for modelling ZCash's upcoming [Trailing Finality Layer](https://electric-coin-company.github.io/tfl-book/).
//!
//! Uses a single master event queue with unified serialization/operational data format for
//! verifying all the nodes in a single network. Intended to allow for authored, regression, fuzz, and
//! randomized testing.
//!
//!
//! ## Good places to start exploring the source code
//! - [`Sim::new`]
//! - [`Sim::tick`]
//! - [`SimNode::handle_msg`]
//! - [`SimNode::tick`]
//!

//! ## Alternative high-level approaches
//!
//! Beyond normal testing, there are multiple approaches to validating/verifying distributed systems.
//! The different options trade off in-depth inspection/confirmation, ability to cause specific
//! (fault/timing) scenarios, coverage of actual production code:
//!
//! - Formal modelling languages/checkers like [TLA+/TLC](https://lamport.azurewebsites.net/tla/high-level-view.html).
//! - **This code:** a simulated simplified model of operations in regular code.
//!   There is a sliding scale of how much production code is used up to...
//! - Deterministic Simulation Testing: using all production code except for simulated sources of non-determinism (network, disk IO, time, ...) to run a full distributed system within a single process.
//!   - [TigerBeetle's "VOPR"](https://docs.tigerbeetle.com/about/vopr/)
//!   - ["Testing Distributed Systems w/ Deterministic Simulation" by Will Wilson](https://www.youtube.com/watch?v=4fFDFbi3toc)
//!   - ["What's the big deal about Deterministic Simulation Testing?"](https://notes.eatonphil.com/2024-08-20-deterministic-simulation-testing.html)
//!   - [`madsim-rs` - "Magical Deterministic Simulator for distributed systems in Rust"](https://github.com/madsim-rs/madsim)
//! - Run full nodes in multiple processes with a TCP/process supervisor ([TigerBeetle Vortex](https://tigerbeetle.com/blog/2025-02-13-a-descent-into-the-vortex/))
//! - Run full nodes in containers within a deterministic hypervisor ([Antithesis](https://antithesis.com/blog/deterministic_hypervisor))
//! - Black-box testing & fault injection across multiple physical nodes ([Jepsen](https://github.com/jepsen-io/jepsen))
//!

//! ## Determinism
//!
//! While non-deterministic testing will still surface bugs or model inconsistencies, determinism
//! is a particularly desirable feature as it means that once we've found an issue we can rerun it
//! as many times as we like to understand it in detail before improving the model and/or fixing the
//! implementation.
//!
//!
//!
//! ### Sources of non-determinism
//!
//! A particularly common source of non-determinism comes from using multiple threads/process on a
//! typical operating system. The scheduling of these is unpredictable and inconsistent.
//! Running the same code multiple times will sometimes have A happen before B, and sometimes B
//! before A.
//!
//! The easiest way to avoid this is to force all operations to occur on a single thread, using
//! other mechanisms to achieve concurrent behaviour.
//!
//! Changing the sequence of events can lead to non-deterministic behaviour at the small scale (e.g.
//! which value got set in a data race), and at the large (e.g. implicitly tie-breaking which
//! element to work with based on which "arrived" first).
//!
//! - Time
//!   - Real-world time
//!   - The duration of time that operations take
//! - Physical infrastructure, disk IO, networks
//!   - Significant source of time disparity
//!   - Data corruption
//!   - Data inaccessibility
//! - Uncontrolled uses of RNG
//!   - Some common operations are not deterministic across runs, notably HashMap iteration.
//!     This can be mitigated with:
//!     - custom hashing functions that aren't randomly seeded
//!     - alternative data structures (e.g. BTreeMap/indexmap)
//!     - a normalization step (e.g. lexicographically sorting) to make externally-visible behaviour deterministic,
//!     although this isn't ideal.
//!

//! ### Cascading/Isolating Changes
//! Small changes cascade, resulting in wildly different subsequent states.
//! There's an open question as to how much you should be able to modify without breaking
//! determinism/stability "guarantees", which will be touched on later.
//!

//! ## Unpredictability
//!
//! The fact that small changes in input sequence/data can result in large changes in execution implies
//! that we should be testing for these. While there may be specific sequences that devs are
//! suspicious of up-front, it's very difficult to manually both conceive of and specify all the
//! possible ways in which concurrent processes might interact.
//!
//! Given that the actual execution of our code is on non-deterministic systems with unpredictable
//! data and sequencing, it seems prudent to check that our code can handle this.
//!
//! Being able to generate unpredictable sequences becomes invaluable, as it allows us to test far
//! more circumstances than we can manually specify. This sounds as though it would be at odds with
//! the desire for determinism, but it is actually distinct. We can achieve both simultaneously.
//!


//! ## Choice of random number generator (RNG)
//!
//! ### "Random"
//! While we typically treat random number generators as just that, random, the ones that we care
//! about here are actually *pseudo-random*: given the same starting state, they will
//! deterministically produce the same sequence of numbers.
//! They are, however, unpredictable: changing a single bit in the input state will (ideally)
//! result in a wildly different sequence.
//! This is exactly what I've just stated we want: unpredictable *and* deterministic.
//!
//! RNGs can be thought of as hashing some internal state, returning the hash, and updating the
//! internal state based on the hash ready for the next iteration.
//! This may happen multiple times if either more bytes are asked for than are in the hash
//! (e.g. `Rng.fill`), or to ensure a particular distribution without bias ([Lemire - "Fast Random Integer Generation in an Interval"](https://arxiv.org/pdf/1805.10941)).
//!
//! The repeated hashing and updating will walk a "random" path through the state space.
//! Note that if/when the state is the same value as any previous value, it will effectively
//! loop from that point, repeating a cycle of values.
//! (Because the output "hash" typically has fewer bits than the state, the same number can be
//! output multiple times without hitting a cycle. This is on top of the range reduction on the
//! hash that is done in most actual use-cases.)
//!
//!
//! ### Seeding
//!
//! Determining the initial state is called "seeding", and given just the seed value we can
//! reliably regenerate the same entire sequence of values.
//!
//! Assuming that there is a (very large!) single cycle for the entire random state, seeding
//! simply chooses where on that random path the generation starts, and continues the same way.
//!
//!
//! ### Streams
//!
//! Inserting a call to generate a random number will affect every subsequent value produced.
//! i.e. random number generation is *highly* order-dependent.
//! In order to minimize the scope of the changes that insertions make, we can have multiple
//! independent work items.
//! In order to remove the dependence between different work items, we initialize multiple
//! different starting states, one per work item.
//! These can be initialized based on a master seed, so they are still deterministic.
//! In fact, this can be done recursively to account for nested scopes.
//!
//! Choosing a different state effectively chooses a different start-point on the same path.
//! While pragmatically this is probably sufficient for test generation, some RNGs have an alternative
//! option in the form of "streams": each stream walks the state through a *different path*.
//!
//!
//! ### Choosing from available options
//!
//! Properties we want:
//! - Fast
//! - Seedable
//! - Deterministic output for given stream/seed across platforms
//!   (This rules out the default `rand` options.)
//! - Bonus feature: ability to select streams: one stream per work item makes them sequence-independent.
//!   This *might* allow for parallel simulation work, if that's not made impossible by other considerations.
//!   This would require a large additional level of confidence that all cross-thread work is necessarily
//!   reproducibly serializable, with a corresponding engineering effort in the simulator that may not be worthwhile.
//!   ([Deterministic Multi-Threaded - Parallel RNGs - The Rust Rand Book](https://rust-random.github.io/book/guide-parallel.html#practice-deterministic-multi-threaded))
//!
//! Viable options:
//! - ChaCha8/20
//!   - Cryptographically secure - more than necessary(?)
//!   - Significantly slower (2-4X) than non-crypto alternatives.
//!   - **Used here for seed generation.**
//! - Xoshiro256++
//!   - There is significant drama between the creators of this & PCG, such that it is difficult
//!     as an outsider to get an objective comparison.
//!   - Used by [`madsim-rs`](https://github.com/madsim-rs/madsim)
//! - PCG family
//!   - Streams *seem* better-supported
//!   - Used for NumPy's RNG
//!   - Specifying the specific version here in case default aliases change.
//!   - DXSM variant has [better cross-stream independence](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)
//!   - **Used here for everything after seed generation.**
//!

//! ## Use cases
//!
//! With this background in mind, it's valuable to have some idea of the ways in which we'd like
//! to test our model/code.
//!
//! Once again there are multiple different alternatives with various trade-offs in terms of input
//! requirements/stability/testing for known uncertainties/testing for complete unknowns:
//!
//! - Random sequences of events.
//! - Coverage-based structured fuzzing of events.
//! - Debugging a specific sequence of events without changing any code: repeatedly running the same seed.
//! - Debugging a *reduced* sequence of events: running a serialized *reduced* requence.
//! - (Regression) Testing: running from a serialized sequence of events (that has previously caused issue), reduced or not.
//! - Manually-authored tests.
//!
//! These would typically be considered as mutually exclusive, each needing a separate approach... but there's no reason they have to be.
//!
//! Structuring our seed input as a pipeline allows for one implementation to enable all of these:
//!
//! |||
//! |-|-|
//! | 1. Generate seed                                     | Random runner starts here |
//! | 2. Use seed to generate entropy source               | Debug rerun with seed starts here |
//! | 3. Entropy source is used to generate init events    | Fuzzer starts here |
//! | 4. Handle init events, creating new events as needed | Testing/sequence debug starts here |
//! | 5. Handle all remaining events                       | ... or here (replaying events without generating anything new) |
//!
//!
//! Because small changes to the code (changing random range extents, adding/removing/reordering
//! generation) can completely alter the sequence, seeds can only be considered "stable" for a
//! given source code, e.g. for a single git commit. Replaying a serialized, structured log of
//! events is required for longer-term stability, which is needed for real-world automated testing.
//!

//! ## Structure & Data Format
//!
//! The requirement of a serializable structured log implies we need a data format for representing this.
//! Fortunately, as our problem domain is highly time-dependent, there is an obvious
//! representation: time-series data in 1 or more arrays, sorted by time.
//!
//! Having a single global notion of time (which is non-trivial in real distributed systems) and
//! a unified sequence of events across all nodes, allows for arbitrary interleaving of operations
//! as well as being a tractable structure for humans to interpret. Different filters on this (e.g. by
//! node/message id) allow for localized views of specific related events.
//!
//! In order to be serializable, all events *need* to be representable as data. Using callbacks/traits/
//! inheritance could be layered on top of this, but this would only be worthwhile if it provided a
//! significantly better abstract model to reason about.
//! (The fact that this is additional to the required code makes this definitionally accidental complexity.)
//! From what I can predict so far, these more "idiomatic" approaches would be a mere restatement
//! of the data format, adding an extra translation requirement without any abstract model improvement.
//! *Authoring* manual event sequences (not operating on them mid-simulation) will likely benefit from some
//! helper functions.
//!
//! There is some acknowledged friction in determining data formats for new event types, but this
//! is necessary for serialization anyway, and provides a prompt to consider how the simulator
//! should test new events.
//!
//! Thinking of the sequence of events as "just data" comes with a major upside: not only can the
//! simulator peek into multiple nodes in the network and confirm their properties correspond with
//! each other as they should, it can also directly assert that the state conforms with expected
//! values derived from sequences of data across time!
//!
//!
//! *The initial implementation focus is on the deterministic random input, keeping in mind the
//! other requirements, however API boundaries are liable to shift when the other use-cases are
//! actually implemented.*

#[macro_use]
extern crate static_assertions;
use rand::{ SeedableRng, Rng, RngCore };
use rand_pcg::Lcg128CmDxsm64 as SimRng;
use rand_chacha::ChaCha8Rng;
use sha2::{ Sha256, Digest };
use sha2::digest::FixedOutput;
use std::mem::size_of;

// NOTE: Thread to pull: time properties
// - Based on simtlfl, we want a discretized notion of time in which multiple events occur (Ticks)
//   as well as a way of totally ordering all events
// - Within ~1.5x of rdtsc freq => allows for:
//   - small-scale order-of-magnitude comparison between time taken for simulation and time simulated.
//   - using known real-world time ranges as initial basis for simulated times.
// - Easy to get seconds out: can use for epoch, which *may* be relevant if there are
//   date/time-based events that could be wrong for e.g. DST changes.
/// Global notion of time. Provides total ordering for events.
#[derive(Debug, Copy, Clone)]
pub struct SimTime { t: u64, }
// struct SimSec { t: u64, } // ALT: bitfields

// impl SimTime {
//     const BITS_PER_S: u8 = 32;
//     fn _get_seconds(&self) -> SimSec {
//         SimSec { t: self.t >> Self::BITS_PER_S }
//     }
// }

type Sha256Hash = [u8; 32];

struct HashAtTime {
    hash: Sha256Hash,
    // time: SimTime, // TODO: use for validation
}


#[derive(Debug, Copy, Clone)]
pub enum Msg {
    BlockMined    { mined_block_hash: Sha256Hash, height: u64 },
    Payload       { payload: u32 },
    VoteValidate  { mined_block_hash: Sha256Hash, prev_final_hash: Sha256Hash },
}

/// Simulation metadata wrapping "real" message.
#[derive(Debug, Copy, Clone)]
pub struct SimMsg {
    src_node_idx : u32,
    dst_node_idx : u32,
    msg          : Msg,
}

// NOTE: an alternative options would be to allow arbitrary callbacks or open inheritance/traits.
//       Keeping this as "just data":
//       - allows for simple (de)serializability
//       - lets the simulator compare data in events to Node state, enabling more types of checks.
//       - adds some friction to creating new event types. The upside it that this provides an
//         opportunity to think through how it should be tested in-sim.

#[derive(Debug, Copy, Clone)]
pub enum EvtKind {
    SimMsg(SimMsg),
    AddNode, // NOTE: if you want to track the node_idx of this on the event, it has to be set on
             // handling this. Otherwise another AddNode event could be inserted beforehand,
             // invalidating the index.
    RemoveNode { node_idx : u32 },
    TickNode   { node_idx : u32 },
}

/// Primary structure for coordinating, serializing & operating on network behaviour.
#[derive(Debug, Copy, Clone)]
pub struct Evt {
    /// absolute global time; not knowable by individual nodes (in reality they
    /// likely have a local time that is similar but not exactly the same)
    pub arrival_time : SimTime,

    evt: EvtKind,
}
const_assert!(size_of::<Evt>() < 128);

/// Root structure for entire network/"universe" simulation
pub struct Sim {
    // Note: could use an additional *stable* priority queue/Eytzinger heap/deque/fixed-size power-of-2 ringbuffer...
    // Currently assuming fairly small number of items after, so constant factors dominate big O.
    do_pos           : bool,
    // seed             : u128, // TODO: may be useful for constructing streams
    finalized_blocks : Vec<HashAtTime>,
    nodes_n          : usize,
    live_nodes_n     : usize,
    at_idx           : usize,
    evts             : Vec<Evt>,
    // Assumption: total number of events in a log will be small enough to fit into memory,
    // otherwise we'd need a fancier streaming read/write system.
}

impl Sim {
    fn log(&self, node_idx: u32, string: &str) {
        let evt_idx = if self.at_idx > 0 { self.at_idx - 1 } else { 0 };
        let time    = self.time();
        println!("{:3} | {:5} | {:4} | {}", evt_idx, time.t, node_idx, string);
    }

    pub fn next_evt(&mut self) -> Evt {
        let i = self.at_idx;
        self.at_idx += 1;
        self.evts[i]
    }

    pub fn push_evt(&mut self, evt: Evt) {
        let mut insert_i = 0;
        for (i, cmp_evt) in (&self.evts).into_iter().enumerate().rev() {
            if cmp_evt.arrival_time.t <= evt.arrival_time.t { // NOTE: items pushed earlier tiebreak to earlier in array
                insert_i = i+1;
                break;
            }

            if i < self.at_idx {
                panic!("trying to insert event into the past; should have inserted by now");
            }
        }

        self.evts.insert(insert_i, evt);
        assert!(self.evts.is_sorted_by_key(|evt| evt.arrival_time.t)); // won't catch instability issues
    }

    /// Schedule the next tick
    pub fn push_tick(&mut self, rng: &mut SimRng, node_idx: u32) {
        self.push_evt(Evt {
            arrival_time: SimTime { t: self.time().t + rng.random_range(1 .. 200) },
            evt: EvtKind::TickNode{ node_idx },
        })
    }

    pub fn time(&self) -> SimTime {
        if self.at_idx == 0 {
            SimTime{ t:0 }
        } else {
            self.evts[self.at_idx-1].arrival_time
        }
    }

    pub fn send_msg(&mut self, rng: &mut SimRng, src_node_idx: u32, dst_node_idx: u32, msg: Msg) {
        self.push_evt(Evt {
            arrival_time: SimTime { t: self.time().t + rng.random_range(1 .. 1000) },
            evt: EvtKind::SimMsg(SimMsg { msg, src_node_idx, dst_node_idx, })
        });
    }

    pub fn send_random_msg(&mut self, rng: &mut SimRng, src_node_idx: u32) {
        if self.nodes_n <= 1 {
            return;
        }

        // send to any node but itself
        let dst_id       = rng.random_range(0 .. self.nodes_n-1) as u32;
        let dst_node_idx = dst_id + (dst_id == src_node_idx) as u32;
        println!("Sending msg from {} to {}", src_node_idx, dst_node_idx);

        let msg = Msg::Payload{ payload: rng.random_range(0 .. 16) };
        self.send_msg(rng, src_node_idx, dst_node_idx, msg);
    }

    pub fn broadcast_msg(&mut self, rng: &mut SimRng, src_node_idx: u32, msg: Msg) {
        self.log(src_node_idx, "broadcasting msg");

        for dst_node_idx in 0 .. self.nodes_n as u32 {
            if dst_node_idx != src_node_idx {
                self.push_evt(Evt {
                    arrival_time: SimTime { t: self.time().t + rng.random_range(100 .. 1000) },
                    evt: EvtKind::SimMsg(SimMsg {
                        msg,
                        src_node_idx,
                        dst_node_idx,
                    })
                });
            }
        }
    }


    pub fn new(maybe_seed: Option<u128>, do_pos: bool) -> (Sim, SimRng, Vec<SimNode>) {
        // NOTE: Seeding (https://rust-random.github.io/book/guide-seeding.html)
        // - Using seed_from_u64 for simplicity.
        //   - Its implementation changing would be treated as a breaking change (https://docs.rs/rand_core/latest/rand_core/trait.SeedableRng.html#method.seed_from_u64)
        //   - Allows for easy repeated testing of multiple scenarios in a loop. (I'm not convinced
        //     this is useful though).
        // - Alternatives include:
        //   - Getting seed from string: allows for "friendly" names.
        //   - Randomized seed: useful for background fuzzing, trying to find scenarios that cause issues.
        //   - Fuzz-like initial byte array acting as consumable source of entropy for seed events. (https://tigerbeetle.com/blog/2023-03-28-random-fuzzy-thoughts/#finite-prng)
        //   - Providing seed directly: which we'll want for repro-ing issues found before.
        let seed : u128 = maybe_seed.unwrap_or_else(||{
            let mut seed_rng = ChaCha8Rng::from_os_rng();
            ((seed_rng.next_u64() as u128) << 64) | seed_rng.next_u64() as u128
        });
        println!("Running with seed {:x}", seed);


        // base/master RNG //
        let mut base_rng = SimRng::new(seed, 0);


        // initial "finalized" block //
        let mut last_finalized_block : [u8; 512] = [0; 512];
        base_rng.fill(&mut last_finalized_block);
        let mut sha256 = Sha256::new();
        sha256.update(&last_finalized_block);
        let finalized_block_hash : Sha256Hash = sha256.finalize_fixed().try_into().unwrap();
        println!("init finalized block hash {}", fmt_sha256_hash(finalized_block_hash));

        let mut finalized_blocks : Vec::<HashAtTime> = Vec::new();
        finalized_blocks.push(HashAtTime { hash: finalized_block_hash }); // time: SimTime { t:0 }


        // main sim structure //
        let mut sim = Sim { do_pos, finalized_blocks, nodes_n: 0, live_nodes_n: 0, at_idx: 0, evts: Vec::new() };


        // nodes backing array //
        let init_nodes_n = base_rng.random_range(2..12); // TODO: allow explicit set
        let nodes : Vec<SimNode> = Vec::with_capacity(init_nodes_n);


        // "seed events" //
        for node_idx in 0..init_nodes_n {
            sim.push_evt(Evt {
                arrival_time: SimTime {t:node_idx as u64 + 1},
                evt: EvtKind::AddNode,
            });
        }

        // top of printed log
        println!("Evt | Time  | Node | Notes");
        println!("--------------------------");
        (sim, base_rng, nodes)
    }

    /// Central message loop handling
    pub fn tick(&mut self, base_rng: &mut SimRng, nodes: &mut Vec<SimNode>) -> bool {
        if ! (self.at_idx < self.evts.len()) {
            return false;
        }

        let evt = self.next_evt();
        let rng = base_rng; // ALT: per node/per event streams to improve independence/decrease cascading changes

        match &evt.evt {
            EvtKind::AddNode => {
                // eliding real initialization & node discovery
                let node_idx = nodes.len();
                self.log(node_idx.try_into().unwrap(), "AddNode");

                let last_finalized_block = &self.finalized_blocks[self.finalized_blocks.len()-1];

                nodes.push(SimNode {
                    node_idx: node_idx.try_into().unwrap(),
                    is_dead: false,
                    node: Node::new(last_finalized_block.hash),
                    // rng: SimRng::new(base_rng.state, node_idx),
                    // q: Vec::new(),
                });

                self.nodes_n      += 1;
                self.live_nodes_n += 1;
                assert!(self.nodes_n == nodes.len());

                nodes[node_idx].tick(rng, self);
            },

            EvtKind::TickNode{ node_idx } => {
                nodes[*node_idx as usize].tick(rng, self);
            },

            EvtKind::RemoveNode{ node_idx } => {
                self.log(*node_idx, "RemoveNode");
                if ! nodes[*node_idx as usize].is_dead {
                    assert!(self.live_nodes_n > 0);
                    self.live_nodes_n -= 1;

                    nodes[*node_idx as usize].is_dead = true;
                }
            }

            EvtKind::SimMsg(msg) => {
                self.log(msg.dst_node_idx, &format!("Receive msg from {}", msg.src_node_idx));
                let dst_node = &mut nodes[msg.dst_node_idx as usize];
                if ! dst_node.is_dead {
                    dst_node.handle_msg(self, rng, msg);
                }
            },
        }

        return true;
    }

    /// Handles events until the queue is empty or an event >= end_time occurs.
    pub fn tick_until(&mut self, rng: &mut SimRng, nodes: &mut Vec<SimNode>, end_time: SimTime) {
        while self.time().t < end_time.t && self.tick(rng, nodes) {
        }
    }

    /// Tick until no unhandled events left in queue.
    pub fn tick_all(&mut self, rng: &mut SimRng, nodes: &mut Vec<SimNode>) {
        while self.tick(rng, nodes) {
        }
    }
}

#[derive(Copy, Clone)]
struct Validation {
    // node_idx:   u32, // TODO: use in verification
    block_hash: Sha256Hash,
}

/// In a full DST, the distinction between the production code for `Node`s and the
/// per-node metadata kept in `SimNode`s is clear. In this model, however, details
/// are elided/abstracted in a way that slightly blurs these lines. I suspect the
/// distinctions should become more clear as more use-cases & events are implemented.
pub struct Node {
    // TODO: q/vec of partial messages/linked-list of partial messages with freelist
    payloads : Vec<u32>,
    last_finalized_block_hash: Sha256Hash,
    best_next_block_hash: Sha256Hash,
    best_next_block_height: u64, // ALT: actual work measure
    needs_validate: bool,
    // TODO: validator priority for tiebreaks
    validations: [Validation; 3],
    validations_n: usize,
}

impl Node {
    fn new(last_finalized_block_hash: Sha256Hash) -> Node {
        Node {
            payloads: Vec::new(),
            last_finalized_block_hash,
            best_next_block_hash   : [0; size_of::<Sha256Hash>()],
            best_next_block_height : 0,
            needs_validate         : false,
            validations            : [Validation{ block_hash: [0; size_of::<Sha256Hash>()] }; 3],
            validations_n          : 0,
        }
    }
}

/// Per-node data that is useful to our simulation/verification, but that would not
/// be visible to "real" nodes.
pub struct SimNode {
    // NOTE: differentiated from Node because this contains global info that actual nodes cannot know.
    node_idx: u32, // TODO: real nodes need some internal ID
    is_dead: bool, // NOTE: sim may still want to inspect data, hence not Option<SimNode>
    // rng: &mut SimRng, // either per-node or global
    node: Node,    // ALT: if runs are small enough, we may want a persistent view of all changes
                   // (at event granularity?)
}

fn fmt_sha256_hash(hash: Sha256Hash) -> String {
    let vals : [u64; 4] = [
        u64::from_le_bytes(hash[ 0.. 8].try_into().unwrap()),
        u64::from_le_bytes(hash[ 8..16].try_into().unwrap()),
        u64::from_le_bytes(hash[16..24].try_into().unwrap()),
        u64::from_le_bytes(hash[24..32].try_into().unwrap()),
    ];
    format!("{:x}_{:x}_{:x}_{:x}", vals[0], vals[1], vals[2], vals[3])
}

impl SimNode {
    /// Performs (non-message-handling) behaviour.
    /// Assumes that this group of *local* node behaviour occurs atomically w.r.t the rest of the network.
    /// May or may not schedule a subsequent tick.
    /// Small chance that node successfully mines a block.
    pub fn tick(&mut self, rng: &mut SimRng, sim: &mut Sim) {
        sim.log(self.node_idx, "TickNode");

        // TODO: shutdown_node
        if rng.random_bool(0.6) {
            sim.push_tick(rng, self.node_idx);
        }

        // TODO: wait for epoch
        if self.node.needs_validate {
            sim.log(self.node_idx, &format!("voting for block (hash: {})", fmt_sha256_hash(self.node.best_next_block_hash)));
            let msg = Msg::VoteValidate {
                mined_block_hash: self.node.best_next_block_hash,
                prev_final_hash:  self.node.last_finalized_block_hash,
            };
            sim.broadcast_msg(rng, self.node_idx, msg);
        }

        if rng.random_bool(0.15) { // mine a new block
            let mut sha256 = Sha256::new();
            sha256.update(&self.node.last_finalized_block_hash);
            sha256.update(&self.node.best_next_block_hash);


            let mut mined_block : [u8; 512] = [0; 512];
            rng.fill(&mut mined_block);
            sha256.update(&mined_block);

            let mined_block_hash : Sha256Hash = sha256.finalize_fixed().try_into().unwrap();
            sim.log(self.node_idx, &format!("block mined (hash: {})", fmt_sha256_hash(mined_block_hash)));
            sim.broadcast_msg(rng, self.node_idx, Msg::BlockMined { mined_block_hash, height: self.node.best_next_block_height + 1 });
        }
    }

    /// Handle messages without buffering. Schedules a subsequent tick.
    // ALT: move all arrived messages into a ringbuffer and handle on tick
    pub fn handle_msg(&mut self, sim: &mut Sim, rng: &mut SimRng, msg: &SimMsg) {
        match &msg.msg {
            Msg::Payload{payload} => {
                // primarily just a demonstration of random API
                self.node.payloads.push(*payload);

                let rand_val = rng.random_range(0.0 ..= 1.0);
                if rand_val < 0.6 {
                    print!("Random msg ");
                    sim.send_random_msg(rng, self.node_idx);
                } else if rand_val < 0.8 {
                    // do something else
                } else {
                    // do nothing
                }
            },

            Msg::BlockMined { mined_block_hash, height } => {
                // TODO: enqueue if highest & comes from last validated, then do validation later on tick
                // TODO: check if follows in sequence from last validated
                if self.node.best_next_block_height < *height {
                    self.node.best_next_block_height = *height;
                    self.node.best_next_block_hash   = *mined_block_hash;
                    if sim.do_pos {
                        self.node.needs_validate = node_is_validator(self.node_idx, sim.nodes_n, &self.node.last_finalized_block_hash);
                    }
                }
            },

            Msg::VoteValidate { mined_block_hash, prev_final_hash } => {
                assert!(sim.do_pos);
                // check if src_node is validator
                // TODO: account for last_finalized_block desync
                let mut is_failed_vote = *prev_final_hash == self.node.last_finalized_block_hash;
                assert!(! is_failed_vote);

                if ! is_failed_vote && node_is_validator(msg.src_node_idx, sim.nodes_n, &self.node.last_finalized_block_hash) {
                    assert!(self.node.validations_n < 3);

                    let mut is_successful_vote = false;
                    // check if 2/3 validations are now the same
                    for i in 0..self.node.validations_n {
                        if self.node.validations[i].block_hash == *mined_block_hash {
                            is_successful_vote = true;
                        }
                    }

                    is_failed_vote = self.node.validations_n == 2;

                    if is_successful_vote || is_failed_vote {
                        if is_successful_vote {
                            sim.log(self.node_idx, &format!("newly-validated: (hash: {})", fmt_sha256_hash(*mined_block_hash)));
                            self.node.last_finalized_block_hash = *mined_block_hash;
                        }

                        self.node.validations   = [Validation{ block_hash: [0; size_of::<Sha256Hash>()] }; 3];
                        self.node.validations_n = 0;
                    } else {
                        self.node.validations[self.node.validations_n] = Validation {
                            // node_idx: msg.src_node_idx,
                            block_hash: *mined_block_hash,
                        };
                        self.node.validations_n += 1;
                    }
                }
            }
        }

        sim.push_tick(rng, self.node_idx); // ALT: iff not currently awaiting tick
    }
}

/// Use last valid block hash to determine which nodes are validators without additional inter-node
/// communication.
fn node_is_validator(node_idx: u32, nodes_n: usize, last_valid_block_hash: &Sha256Hash) -> bool {
    let vals : [u64; 4] = [
        u64::from_le_bytes(last_valid_block_hash[ 0.. 8].try_into().unwrap()),
        u64::from_le_bytes(last_valid_block_hash[ 8..16].try_into().unwrap()),
        u64::from_le_bytes(last_valid_block_hash[16..24].try_into().unwrap()),
        u64::from_le_bytes(last_valid_block_hash[24..32].try_into().unwrap()),
    ];

    // TODO: account for non-uniform stake
    // TODO: nodes should all get the same result regardless of known nodes_n
    let validators : [u32; 3] = [
        (vals[0]             % nodes_n as u64).try_into().unwrap(),
        (vals[1]             % nodes_n as u64).try_into().unwrap(),
        ((vals[2] ^ vals[3]) % nodes_n as u64).try_into().unwrap(),
    ];

    node_idx == validators[0] ||
        node_idx == validators[1] ||
        node_idx == validators[2]
}



#[cfg(test)]
mod tests {
    use super::*;
    // NOTE: these aren't great seeds for real
    // NOTE: run with `cargo test -- --show-output`

    #[test]
    fn test1() {
        let (mut sim, mut base_rng, mut nodes) = Sim::new(Some(123), false);
        sim.tick_all(&mut base_rng, &mut nodes);

        // NOTE: a quick check for reproducibility is to run multiple times with the same input,
        //       generate the next random number for each stream and check that they are the same.
    }

    #[test]
    #[should_panic]
    fn test2() {
        let (mut sim, mut base_rng, mut nodes) = Sim::new(Some(123), true);
        sim.tick_all(&mut base_rng, &mut nodes);
    }

    #[test]
    fn test3() {
        let (mut sim, mut base_rng, mut nodes) = Sim::new(Some(122), false);
        sim.tick_all(&mut base_rng, &mut nodes);
    }

    #[test]
    fn test4() {
        let (mut sim, mut base_rng, mut nodes) = Sim::new(Some(234), false);
        sim.tick_until(&mut base_rng, &mut nodes, SimTime { t: 10000 });
    }
}
