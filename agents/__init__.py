from agents.affine_psm import AffinePSMAgent
from agents.fb import FBAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.latent_affine_psm import LatentAffinePSMAgent
from agents.latentrl import LatentRLAgent
from agents.psm import PSMAgent
from agents.psmflow import PSMFlowAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    affine_psm=AffinePSMAgent,
    fb=FBAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    latent_affine_psm=LatentAffinePSMAgent,
    latentrl=LatentRLAgent,
    psm=PSMAgent,
    psmflow=PSMFlowAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
