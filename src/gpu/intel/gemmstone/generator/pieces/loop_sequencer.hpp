/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GEMMSTONE_GENERATOR_PIECES_LOOP_SEQUENCER_HPP
#define GEMMSTONE_GENERATOR_PIECES_LOOP_SEQUENCER_HPP

#include <array>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "gemmstone/config.hpp"

GEMMSTONE_NAMESPACE_START

namespace loop_sequencer {

class Iteration {
    friend class LoopSequencer;

public:
    // Iteration counter.
    constexpr operator int()      const { return value; }
    constexpr int get()           const { return value; }
    // # of guaranteed iterations after this one
    constexpr int remaining()     const { return rem; }
    // Offset between current counter value and the iteration when the action triggers.
    constexpr int counterOffset() const { return offset; }

    Iteration() {}

private:
    Iteration(int value_, int rem_, int offset_) : value(value_), rem(rem_), offset(offset_) {}

    int value;
    int rem;
    int offset;
};

class LoopSequencer {
public:
    struct Requirements {
        int period = 0;         // # of loops between repetitions of the action.
        int phase = 0;          // Action triggers when (loop #) mod period = phase.
        int duration = 0;       // Action only triggered if at least _duration_ loops remain (including current loop).
        int rduration = 0;      // In reverse mode, action triggers at most _rduration_ loops after the trigger loop (inclusive).
        int lookahead = 0;      // Action will run _lookahead_ loops before trigger condition.
        int variants = 0;       // # of variants of the action.
                                //   variants = n is equivalent to scheduling n copies of the action at n times the period, with equally spaced phases.
        int checkType = 0;      // See CheckType enum.

        friend Requirements operator|(const Requirements &req1, const Requirements &req2) {
            auto result = req1;
            result.period    |= req2.period;
            result.phase     |= req2.phase;
            result.duration  |= req2.duration;
            result.rduration |= req2.rduration;
            result.lookahead |= req2.lookahead;
            result.variants  |= req2.variants;
            result.checkType |= req2.checkType;
            return result;
        }

        Requirements delay(int delay_) const {
            auto result = *this;
            result.phase += delay_;
            result.duration -= delay_;
            result.duration = std::max(result.duration, 0);
            return result;
        }
    };

    using ActionFunc = std::function<void (Iteration)>;         // void action(Iteration iteration);
    using ActionCheckFunc = std::function<bool (Iteration)>;    // bool actionCheck(Iteration iteration);
    using Callback = std::function<void (int, int)>;            // void callback(int arg1, int arg2);

    struct Item {
        Requirements req;
        ActionFunc action;

        Item(Requirements req_, ActionFunc action_) : req(req_), action(std::move(action_)) {}
    };

    struct CheckedItem : public Item {
        ActionCheckFunc check;

        CheckedItem(Requirements req_, ActionFunc action_, ActionCheckFunc check_) : Item(req_, std::move(action_)), check(std::move(check_)) {}
        CheckedItem(Item item_, ActionCheckFunc check_ = ActionCheckFunc(nullptr)) : Item(std::move(item_)), check(std::move(check_)) {}
    };

    enum class CallbackType {
        OffsetCounter,      // Add offset to loop counter.                                  (arg1 = offset)
        LoopStart,          // Mark top of loop. Jump to bottom if counter <= 0.            (arg1 = unroll)
        LoopEnd,            // Decrement counter by unroll and jump to top of loop if > 0.  (arg1 = unroll)
        Jump,               // Jump unconditionally.                                        (arg1 = label)
        JumpIfLT,           // Jump if counter < arg.                                       (arg1 = threshold, arg2 = label)
        JumpTarget,         // Mark jump target.                                            (arg1 = label)
        NotifyPhase,        // Notify of change in phase.                                   (arg1 = phase)
        _end_
    };

    enum class RemainderHandling {
        Separate,           // Full remainder handling for both main and cooldown loops.
        Unified,            // Combine main and cooldown loop remainder handling.
        Ignore              // No remainder handling; assume loop count is multiple of unroll.
    };

    enum Phase {
        PhaseWarmup,        // Warmup for main loop.
        PhaseMainLoop,      // Inside main loop.
        PhaseCooldown,      // Cooldown after main loop.
        PhaseMainPathEnd,   // End of main path.
        PhaseShortLoop,     // Short loop, if not enough iterations for main loop.
        PhaseShortLoopEnd,  // End of short loop.
        PhaseRemainder,     // Unified remainder loop.
        PhaseFullyUnrolled, // Fully unrolled loop sequence.
    };

    enum CheckType {
        StandardCheck = 0,  // Loop counter should be checked to see if action can be run.
        OptionalCheck = 1,  // Loop counter may be checked to see if action can be run, but it is not required.
        Unconditional = 3,  // Loop counter must not be checked to see if action can be run.
    };

    void schedule(Requirements reqs, ActionFunc action);
    void schedule(std::vector<Item> list);
    void schedule_if(Requirements reqs, ActionFunc action, ActionCheckFunc check);
    void schedule_if(std::vector<CheckedItem> list);
    void swapLast2();
    void analyze();
    void materialize(int minLoops = 0, int maxLoops = -1);

    void setCallback(CallbackType type, Callback cb);
    void setRemainderHandling(RemainderHandling handling);
    void setReverse(bool reverse_ = true) { reverse = reverse_; }
    void counterIsMultipleOf(int align) { counterAlign = align; }

    int getUnroll() const;
    int getWarmup() const;
    int getCooldown() const;
    int getLoopBias() const;

protected:
    struct Action {
        std::vector<CheckedItem> list;
        int nextTrigger;
    };

    enum {
        NeverScheduled = std::numeric_limits<int>::min()
    };

    std::vector<Action> actions;
    std::array<Callback, static_cast<size_t>(CallbackType::_end_)> callbacks;
    std::vector<std::pair<int, int>> activeChecks;
    RemainderHandling remainderHandling = RemainderHandling::Separate;
    int nextLabel = 0;
    int currentBias = 0;

    int unroll = 1;
    int maxLookahead = 0;
    int minCooldown = 0;
    int warmup = 0;
    int counterAlign = 1;

    bool reverse = false;
    bool analyzed = false;

    void validate(std::vector<CheckedItem> &list);
    void callback(CallbackType type, int arg1, int arg2 = 0);
    void run(int l, int guaranteedMin, int guaranteedMax = -1, int alignOffset = 0);
    void closeChecks();
    bool precheck(int thresh);
    void resetActions();
    void adjustActionTriggers(int shift);
    void checkAnalyzed() const;
};


static inline LoopSequencer::Requirements every(int period)
{
    LoopSequencer::Requirements result;
    result.period = period;
    return result;
}

static inline LoopSequencer::Requirements every(int ph, int period)
{
    LoopSequencer::Requirements result;
    result.phase = ph;
    result.period = period;
    return result;
}

static inline LoopSequencer::Requirements phase(int ph)
{
    LoopSequencer::Requirements result;
    result.phase = ph;
    return result;
}

static inline LoopSequencer::Requirements duration(int dur)
{
    LoopSequencer::Requirements result;
    result.duration = dur;
    return result;
}

static inline LoopSequencer::Requirements rduration(int rdur)
{
    LoopSequencer::Requirements result;
    result.rduration = rdur;
    return result;
}

static inline LoopSequencer::Requirements lookahead(int ahead)
{
    LoopSequencer::Requirements result;
    result.lookahead = ahead;
    return result;
}

static inline LoopSequencer::Requirements variants(int vars)
{
    LoopSequencer::Requirements result;
    result.variants = vars;
    return result;
}

static inline LoopSequencer::Requirements unconditional()
{
    LoopSequencer::Requirements result;
    result.checkType = LoopSequencer::Unconditional;
    return result;
}

static inline LoopSequencer::Requirements checkOptional()
{
    LoopSequencer::Requirements result;
    result.checkType = LoopSequencer::OptionalCheck;
    return result;
}

} /* namespace loop_sequencer */

using loop_sequencer::LoopSequencer;

GEMMSTONE_NAMESPACE_END

#endif /* GEMMSTONE_GENERATOR_PIECES_LOOP_SEQUENCER_HPP */
