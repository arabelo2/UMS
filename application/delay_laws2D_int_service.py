# application/delay_laws2D_int_service.py

from domain.delay_laws2D_int import DelayLaws2DInt

class DelayLaws2DIntService:
    """
    Service to compute delay laws for an array at a fluid/fluid interface.
    """
    def compute_delays(self, M, s, angt, ang20, DT0, DF, c1, c2, plt_option='n'):
        delay_obj = DelayLaws2DInt(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)
        return delay_obj.compute_delays()

def run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, plt_option='n'):
    service = DelayLaws2DIntService()
    return service.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)
