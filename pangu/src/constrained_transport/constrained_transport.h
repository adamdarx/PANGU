// Copyright (c) 2026 Yuehang Li.
// Licensed under the MIT License. See LICENSE in the project root for license information.

#ifndef PANGU_SRC_CONSTRAINEDTRANSPORT_CONSTRAINEDTRANSPORT_H
#define PANGU_SRC_CONSTRAINEDTRANSPORT_CONSTRAINEDTRANSPORT_H

#include <memory>
#include <parthenon/package.hpp>

// Updates electric fields and magnetic fluxes with constrained transport.
parthenon::TaskStatus ConstraintedTransport(
    std::shared_ptr<parthenon::MeshBlockData<parthenon::Real>> &resource);

#endif  // PANGU_SRC_CONSTRAINEDTRANSPORT_CONSTRAINEDTRANSPORT_H
