import aasma.agent as agent

from randomVsRandom import EpisodeInfo

class MemoryAgent(agent.Agent):
    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment, episodeInfo: EpisodeInfo, wantsToReceiveInformation = False):
        super().__init__(agentId, nSeekers, nPreys, is_prey, environment, wantsToReceiveInformation)

        self.episodeInfo = episodeInfo

    def action(self) -> int:
        step = self.environment._step_count

        if self.is_prey():
            offset = step * self.nPreys + self.agentId
            return self.episodeInfo.actions_prey[offset]
        else:
            offset = step * self.nSeekers + self.agentId
            return self.episodeInfo.actions_seekers[offset]
