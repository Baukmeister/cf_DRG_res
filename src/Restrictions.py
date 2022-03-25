from enum import Enum


class RestrictionKind(Enum):
    DRUG_DRUG = 0,
    DRUG_DISEASE = 1


class Restriction:

    def __init__(self, kind: RestrictionKind, itemA: str, itemB: str):
        self.kind = kind
        self.itemA = itemA
        self.itemB = itemB

    def get_compliant_sequence_indices(self, event_sequences, diagnoses):

        if self.kind is RestrictionKind.DRUG_DRUG:

            compliant_indices = [i for i, seq in enumerate(event_sequences)
                                 if not (self.itemA in seq and self.itemB in seq)]

        elif self.kind is RestrictionKind.DRUG_DISEASE:

            compliant_indices = [i for i, (drugs, diseases) in enumerate(zip(event_sequences, diagnoses))
                                 if not (self.itemA in drugs and self.itemB in diseases)]

        print(f"While enforcing restriction {len(compliant_indices)} out of {len(event_sequences)} were kept")

        return compliant_indices
