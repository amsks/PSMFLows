from agents.fb import FBAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.psm import PSMAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fb=FBAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    psm=PSMAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
