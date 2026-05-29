# Treadmill Energy Dashboard

Do you have a split-belt instrumented treadmill, and you are doing metabolics experiments, but waiting for mocap results is tedious and awful?

This is a set of tools to get live insights out of your force-instrumented treadmill. If you have a split-belt treadmill with separate force plates under each foot, you can continuously estimate:
- COM velocity and acceleration, subject mass, and tiny force plate angular calibration errors
- Power rates per foot, by dotting force with COM velocity
- Stride length, current percent of stride, stride frequency

From this, you can produce plots of when and how much energy is entering and leaving the body plotted along the stride. Over flat ground constant speed cyclic gait the net energy is guaranteed to sum to 0 (otherwise it would not be a cycle). This means you can reason about the impact of changes to gait in one place in the stride creating "waterbed" changes to gait in another place in the stride.

## Relationship to Metabolics

A bicycle rolling along a treadmill at constant speed would show a 0-power rate over an entire "stride". The mystery this dashboard attempts to visualize in realtime is exactly why human gait is not as efficient as a bicycle.

### Inverted Pendulum Models

In theory, the rolling motion of the rigid inverted pendulum models for human gait should also be 0-power (because the pendulum is always moving exactly perpendicular to ground force), except for a large non-physical instantaneous impulse spike when we change between pendulums. Art Kuo and friends have written a number of papers theorizing that a lot of the metabolic cost of walking comes from these heelstrike transition losses.

### Spring Loaded Inverted Pendulum Models

A spring-loaded inverted pendulum (SLIP) model does away with this huge non-physical instantaneous impulse, in favor of instead having non-zero power between the ground and the walker, which is stored and then released in the spring leg. Now the dynamics of the system are smooth and differentiable, and the resulting model ground force curves look more like what we actually see from human gait, with the characteristic "double hump" shape. Without friction, the SLIP model says that walking should require no net propulsive power. We simply store and release energy in each leg in a perfectly passive manner as we roll forwards. This is obviously not how the biological leg actually works, but raises interesting design questions and objectives for exoskeletons that are looking to improve energy economy.

### Real Tendon Elastic Energy Storage

Biologically, elastic energy storage in the unassisted human leg requires efficient tendon loading. The main tendon for energy storage is the Achilles, which requires that the ankle be dorsiflexed, which means that it cannot load right at heelstrike. Other tendons are extremely stiff (presumably in order for evolution to allow good position control bandwidth from the muscles to the joints). Crucially this includes the tendon connecting the patella (kneecap) to the tibia (shin) which allows your quads to keep your knees from buckling. At heelstrike, the spots where elastic energy storage can happen include this very stiff tendon, and also elastic foot deformation.

In practice, the negative work at heelstrike seems to actually be muscles doing negative work (eccentric contraction), both in the quads (slowly allowing the knee to bend) and in the ankle dorsiflexors (preventing "foot slap"). Because net energy over a stride must sum to 0, this requires that positive work be done later.

### Exoskeleton Assistance

If the user is wearing an exoskeleton, that exoskeleton will likely have joint encoders and force/torque sensors, where it will be possible to estimate the instantaneous power of the exoskeleton. If exoskeleton power is available, then we have the power rates at both "ports" to the human: the treadmill, and the exoskeleton. That means that we can compute the internal power of the human. We know that the integral of the human + exoskeleton power over a stride must equal 0 (assuming flat ground, fixed pace). That implies that net positive power by an exoskeleton is going to force net negative power from a human, and vice versa.

## Identifying which Joints Are Active

There are two reasons we want to estimate joint angles. For energy estimates, it is useful because in order to source or sink power at a joint, it must be moving. For static torque estimates, knowing the joint centers in relation to the CoP data allows us to compute cross products.

The mystery of Achillese tendon energy flow (which is less metabolically expensive than other kinds of energy flow) can be partially constrained by knowing that Achillese storage requires active ankle dorsiflexion velocity, and return requires plantarflexion velocity. That limits the times that the Achillese can be responsible for observed energy flow.

To do this, we must estimate the joint's velocity. There are several ways to do this, with varying accuracy.

### Phase Based Estimates

We can just assume that the user's gait is "standard", and estimate joint velocities based on phases.

### Lookup Table Estimates

Better than phases, we can lookup gait cycles based on their GRF/CoP pattern similarity to the AddBiomechanics data, and estimate joint velocities based on that.

### Joint Encoders

If exoskeletons come with joint encoders, we can use those to fine-tune estimates for particular joints, and also to refine our lookup table estimates above.

### Mocap

If we have live mocap (from scaled triads with distance signatures to ID them), we can of course use that.

## Methods to Lower Bound Metabolic Cost (mhich are Real-Time and Interpretable)

We have several options to lower bound metabolic cost.

As a rough unit conversion, we will say that:
- 1 Joule of negative mechanical work done in a muscle costs 1 Joule of chemical energy
- 1 Joule of positive mechanical work done in a muscle costs 4 Joules of chemical energy

We know that energy into and out of tendons over the course of a stride must integrate to 0. On its face, this does not help us, because the energy into and out of the human + exo system over a stride must also integrate to 0. However, if we can identify some section of energy flow as certainly _not_ having come from tendons, and we can still bound the total energy flow in the tendons as integrating to 0, we can infer that there was corresponding balancing energy flow from elsewhere in the system. That balancing energy either came from the human or the exo. If it came from the human, we can use the above rough conversion to estimate the metabolic cost of the work.

### Direct conservative lower bound: just heel strike losses + replacements

We can assume that net negative human work (after subtracting out exo work) at/after heel strike and before dorsiflexion (estimate this as first ~15% of gait cycle) is negative muscle work. Then use the "tendon balance" method to estimate the other necessary muscle work that must be happening to balance the energy flow.

### Alternative conservative lower bound: immobilize the ankles (wear an ankle boot), and assume no tendon storage

If there is no Achilles tendon storage, because the ankle is experimentally immobilized, then we can assume that all positive and negative net human work (after subtracting out exo work) is coming from muscles, and compute a metabolic lower bound.

### Messy unitless non-linear lower bound: implied biological joint torques

We know that static muscle contractions cost metabolic energy. While the exact mapping between static loads and metabolic costs is messy, we know there is a general relationship between biological joint torques (at least at the knees and hips) and muscle contractions. The ankle is a more complicated story, because of the Achilles tendon.

All we need is a safeguard lower bound metric to prevent the dashboard from incorrectly estimating that something like "crouch gait" which requires large static holding torques at the joints could be energentically efficient, just because there is little power flowing in and out of the human. While computing full dynamic torques at each joint requires a detailed physical simulation, which requires full motion capture and fitting a detailed dynamics model of the subject, we do not actually need the full dynamic torques. We already capture the power rates at the joints with the treadmill lower bounds.

To develop our "static muscle contraction" lower-bound metric, we will simply use the GRF crossed with the moment arm from the CoP to the joint center, taking a sum over magnitude of torques at both knees and both hips. This is a lower bound, because the conditional only runs one direction: If this quantity is not zero, we can say that muscle contraction is required at the hips or knees. If it is zero, we cannot necessarily say that the muscles are not co-contracting and wasting energy anyways.

Doing this requires placing marker triads at known locations relative to the joint centers, so this is an optional metric, but is nice to have.

### Joint lower-bound optimization: minimize the energy lower bound and the joint torque lower bound, and let the human relax over time

We cannot control co-contraction and heart rate changes and other metabolic cost contributors from stress and effort. However, we can directly understand and minimize the contributions to metabolic cost from muscle power and non-tendon static torque sources, which we treat as lower bounds on total user effort.