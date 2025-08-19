"""
QuantLib Skills for Advanced Financial Calculations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from decimal import Decimal

# Import QuantLib with fallback
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    ql = None

logger = logging.getLogger(__name__)


class QuantLibSkills:
    """Advanced financial calculations using QuantLib"""
    
    def __init__(self):
        self.calendar = ql.UnitedStates() if QUANTLIB_AVAILABLE else None
        self.day_count = ql.Actual365Fixed() if QUANTLIB_AVAILABLE else None
        
    def price_bond(self, face_value: float, coupon_rate: float, maturity_years: int, 
                   yield_rate: float, frequency: int = 2) -> Dict[str, Any]:
        """Price a fixed-rate bond"""
        if not QUANTLIB_AVAILABLE:
            raise ValueError("QuantLib is required for bond pricing")
        
        try:
            # Set evaluation date
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            # Create bond schedule
            issue_date = today
            maturity_date = today + ql.Period(maturity_years, ql.Years)
            
            schedule = ql.Schedule(
                issue_date,
                maturity_date,
                ql.Period(ql.Semiannual if frequency == 2 else ql.Annual),
                self.calendar,
                ql.Unadjusted,
                ql.Unadjusted,
                ql.DateGeneration.Backward,
                False
            )
            
            # Create bond
            bond = ql.FixedRateBond(
                2,  # Settlement days
                face_value,
                schedule,
                [coupon_rate],
                self.day_count
            )
            
            # Price bond
            yield_curve = ql.FlatForward(today, yield_rate, self.day_count)
            bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))
            bond.setPricingEngine(bond_engine)
            
            price = bond.NPV()
            clean_price = bond.cleanPrice()
            dirty_price = bond.dirtyPrice()
            accrued_interest = bond.accruedAmount()
            ytm = bond.bondYield(clean_price, self.day_count, ql.Compounded, ql.Semiannual)
            duration = ql.BondFunctions.duration(
                bond, yield_rate, self.day_count, ql.Compounded, ql.Semiannual
            )
            convexity = ql.BondFunctions.convexity(
                bond, yield_rate, self.day_count, ql.Compounded, ql.Semiannual
            )
            
            return {
                "face_value": face_value,
                "coupon_rate": coupon_rate,
                "maturity_years": maturity_years,
                "yield_rate": yield_rate,
                "price": price,
                "clean_price": clean_price,
                "dirty_price": dirty_price,
                "accrued_interest": accrued_interest,
                "yield_to_maturity": ytm,
                "duration": duration,
                "convexity": convexity
            }
            
        except Exception as e:
            logger.error(f"Bond pricing failed: {e}")
            raise ValueError(f"Bond pricing failed: {str(e)}")
    
    def price_option(self, option_type: str, spot: float, strike: float, 
                    maturity_days: int, risk_free_rate: float, volatility: float,
                    dividend_yield: float = 0.0) -> Dict[str, Any]:
        """Price European or American option using Black-Scholes"""
        if not QUANTLIB_AVAILABLE:
            raise ValueError("QuantLib is required for option pricing")
        
        try:
            # Set evaluation date
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            # Option parameters
            maturity_date = today + maturity_days
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            
            # Create option
            if option_type.upper() == "CALL":
                payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
            elif option_type.upper() == "PUT":
                payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)
            else:
                raise ValueError(f"Unknown option type: {option_type}")
            
            # European exercise
            european_exercise = ql.EuropeanExercise(maturity_date)
            european_option = ql.VanillaOption(payoff, european_exercise)
            
            # Market data
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, risk_free_rate, self.day_count)
            )
            dividend_yield_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(today, dividend_yield, self.day_count)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, self.calendar, volatility, self.day_count)
            )
            
            # Black-Scholes process
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle,
                dividend_yield_ts,
                flat_ts,
                flat_vol_ts
            )
            
            # Pricing engine
            european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
            
            # Calculate results
            price = european_option.NPV()
            delta = european_option.delta()
            gamma = european_option.gamma()
            vega = european_option.vega() / 100  # Per 1% change in volatility
            theta = european_option.theta() / 365  # Per day
            rho = european_option.rho() / 100  # Per 1% change in rate
            
            # Calculate implied volatility if price is given
            impl_vol = None
            try:
                impl_vol = european_option.impliedVolatility(price, bsm_process)
            except:
                pass
            
            return {
                "option_type": option_type,
                "spot": spot,
                "strike": strike,
                "maturity_days": maturity_days,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "dividend_yield": dividend_yield,
                "price": price,
                "greeks": {
                    "delta": delta,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "rho": rho
                },
                "implied_volatility": impl_vol
            }
            
        except Exception as e:
            logger.error(f"Option pricing failed: {e}")
            raise ValueError(f"Option pricing failed: {str(e)}")
    
    def calculate_var(self, portfolio_values: List[float], confidence_level: float = 0.95,
                     time_horizon: int = 1) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        if not QUANTLIB_AVAILABLE:
            raise ValueError("QuantLib is required for VaR calculation")
        
        try:
            import numpy as np
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Sort returns
            sorted_returns = np.sort(returns)
            
            # Calculate VaR
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[var_index] * np.sqrt(time_horizon)
            
            # Calculate CVaR (Expected Shortfall)
            cvar = -np.mean(sorted_returns[:var_index]) * np.sqrt(time_horizon)
            
            # Calculate statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
            
            return {
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "var": var,
                "cvar": cvar,
                "mean_return": mean_return,
                "volatility": std_return,
                "sharpe_ratio": sharpe_ratio,
                "num_observations": len(returns)
            }
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise ValueError(f"VaR calculation failed: {str(e)}")
    
    def price_swap(self, notional: float, fixed_rate: float, floating_spread: float,
                  maturity_years: int, payment_frequency: int = 4) -> Dict[str, Any]:
        """Price an interest rate swap"""
        if not QUANTLIB_AVAILABLE:
            raise ValueError("QuantLib is required for swap pricing")
        
        try:
            # Set evaluation date
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            # Market data - simplified flat curve
            rate = 0.03  # 3% flat rate
            flat_curve = ql.FlatForward(today, rate, self.day_count)
            curve_handle = ql.YieldTermStructureHandle(flat_curve)
            
            # Swap parameters
            maturity_date = today + ql.Period(maturity_years, ql.Years)
            
            # Fixed leg schedule
            fixed_schedule = ql.Schedule(
                today,
                maturity_date,
                ql.Period(12 // payment_frequency, ql.Months),
                self.calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Forward,
                False
            )
            
            # Floating leg schedule (quarterly)
            float_schedule = ql.Schedule(
                today,
                maturity_date,
                ql.Period(3, ql.Months),
                self.calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Forward,
                False
            )
            
            # Create swap
            index = ql.Euribor3M(curve_handle)
            swap = ql.VanillaSwap(
                ql.VanillaSwap.Payer,
                notional,
                fixed_schedule,
                fixed_rate,
                self.day_count,
                float_schedule,
                index,
                floating_spread,
                self.day_count
            )
            
            # Pricing engine
            swap_engine = ql.DiscountingSwapEngine(curve_handle)
            swap.setPricingEngine(swap_engine)
            
            # Calculate results
            npv = swap.NPV()
            fair_rate = swap.fairRate()
            fixed_leg_npv = swap.fixedLegNPV()
            floating_leg_npv = swap.floatingLegNPV()
            
            return {
                "notional": notional,
                "fixed_rate": fixed_rate,
                "floating_spread": floating_spread,
                "maturity_years": maturity_years,
                "npv": npv,
                "fair_rate": fair_rate,
                "fixed_leg_npv": fixed_leg_npv,
                "floating_leg_npv": floating_leg_npv,
                "payment_frequency": payment_frequency
            }
            
        except Exception as e:
            logger.error(f"Swap pricing failed: {e}")
            raise ValueError(f"Swap pricing failed: {str(e)}")
    
    def calculate_credit_risk(self, notional: float, probability_default: float,
                            recovery_rate: float, maturity_years: int) -> Dict[str, Any]:
        """Calculate credit risk metrics"""
        try:
            # Expected Loss
            loss_given_default = notional * (1 - recovery_rate)
            expected_loss = probability_default * loss_given_default
            
            # Credit VaR (simplified)
            unexpected_loss = loss_given_default * (probability_default * (1 - probability_default)) ** 0.5
            
            # Annualized metrics
            annual_pd = 1 - (1 - probability_default) ** (1 / maturity_years)
            annual_expected_loss = expected_loss / maturity_years
            
            return {
                "notional": notional,
                "probability_default": probability_default,
                "recovery_rate": recovery_rate,
                "maturity_years": maturity_years,
                "loss_given_default": loss_given_default,
                "expected_loss": expected_loss,
                "unexpected_loss": unexpected_loss,
                "annual_probability_default": annual_pd,
                "annual_expected_loss": annual_expected_loss
            }
            
        except Exception as e:
            logger.error(f"Credit risk calculation failed: {e}")
            raise ValueError(f"Credit risk calculation failed: {str(e)}")