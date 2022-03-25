from enum import Enum


class RestrictionKind(Enum):
    DRUG_DRUG = 0,
    DRUG_DISEASE = 1


class Restriction:

    def __init__(self, kind: RestrictionKind, itemA: str, itemB: str):
        self.kind = kind
        self.itemA = itemA
        self.itemB = itemB

    def get_compliant_sequence_indices(self, event_sequences, diagnoses, relevant_indices):

        if self.kind is RestrictionKind.DRUG_DRUG:

            compliant_indices = [i for i in relevant_indices
                                 if not (self.itemA in event_sequences[i] and self.itemB in event_sequences[i])]

        elif self.kind is RestrictionKind.DRUG_DISEASE:

            compliant_indices = [i for i in relevant_indices
                                 if not (self.itemA in event_sequences[i] and self.itemB in diagnoses[i])]

        print(f"While enforcing restriction {len(compliant_indices)} out of {len(event_sequences)} were kept")

        return compliant_indices
