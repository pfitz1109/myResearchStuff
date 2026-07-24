-- -------------------------------------------------------------------*- Lua -*-
-- Copyright (c) 2018-2021
-- University of Notre Dame
-- University of Washington
--
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--
-- * Redistributions of source code must retain the above copyright notice, this
--   list of conditions and the following disclaimer.
--
-- * Redistributions in binary form must reproduce the above copyright notice,
--   this list of conditions and the following disclaimer in the documentation
--   and/or other materials provided with the distribution.
--
-- * Neither the name of the copyright holder nor the names of its
--   contributors may be used to endorse or promote products derived from
--   this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
-- FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
-- DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
-- SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
-- CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
-- OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-- -----------------------------------------------------------------------------
-- Input parameters for the MRWT framework
-- Wavelet parameters
p = 6               -- Wavelet basis
J = 6              -- Maximum resolution level
error = 1e-2        -- Error threshold

-- Spatial parameters
x = {-1, 1}         -- x-interval
y = { 0, 0.5 }         -- y-interval (t-interval)

-- Some function
function density0(x,y)
   return 1.16 -- [kg/m^3] undisturbed density
end

function velocity0(x,y)
   return 0.0, 0.0 -- [m/s] undisturbed velocity
end

function pressure0(x,y)
   local atmP  = 1e5 -- [Pa] undisturbed pressure
   local ratio = 20
   local delta = 1e-2
   local x0    = 0
   local y0    = -0.5
   return atmP*(1 + ratio*math.exp((-(x - x0)^2 - (y - y0)^2)/delta))
end

function energy0(x,y)
         if x < 0.5 then return 0 end
   local gamma = 1.4
   return pressure0(x,y)/((gamma - 1.0)*density0(x,y))
end

function f(x,y)
   return energy0(x,y)
end

function exSol(x,y,A,sigma,x0,nu)
   return sqrt(pow(A*sigma,2)/(2*nu*t+pow(sigma,2)))*exp(-pow(x-x0,2)/(2*(2*nu*t+pow(sigma,2))));
end

function t0Bound(x,y)
   return exSol(x,y,A,sigma,x0,nu)
end

function xLbound(x,y)
   return exSol(x,y,A,sigma,x0,nu)
end

function xRbound(x,y)
   return exSol(x,y,A,sigma,x0,nu)
end
